"""Batched inference primitives shared by estimator families."""

import numpy as np

from ._session import Session


class InferenceRuntime:
    def linear(self, x, weights, bias, *, mean=None):
        # A one-layer score/projection needs no activation workspace or chunk
        # loop. Center while loading tiles instead of writing a centered array.
        n, d = x.shape
        c = weights.shape[1]
        if (c <= 8 and mean is None) or c > 32 or x.nbytes + n * c * 4 > 32 * 1024 * 1024:
            return self.dense_layers(x, [weights], [bias], ["identity"], input_mean=mean)
        with Session(self) as session:
            xb = session.upload(x)
            wb = session.upload(np.ascontiguousarray(weights, dtype=np.float32))
            bb = session.upload(np.broadcast_to(bias, (c,)).astype(np.float32))
            result = session.empty((n, c))
            command = session.command()
            means = wb
            if mean is not None:
                high = np.asarray(mean, np.float32)
                means = session.upload(
                    np.concatenate([high, np.asarray(mean) - high]).astype(np.float32)
                )
            session.encode(
                command,
                "projection_tiled",
                [xb, wb, bb, means, result],
                [n, d, c, int(mean is not None)],
                (c, n),
            )
            session.submit(command)
            return result.read()

    def dense_layers(self, x, weights, biases, activations, *, input_mean=None):
        codes = {"identity": 0, "relu": 1, "tanh": 2, "logistic": 3, "softmax": 4, "exp": 5}
        # Bound temporary activations, including all layers, to about 32 MiB.
        rows = max(1, (32 * 1024 * 1024) // (4 * (x.shape[1] + sum(w.shape[1] for w in weights))))
        output = np.empty((len(x), weights[-1].shape[1]), np.float32)
        with Session(self) as session:
            centering = None
            if input_mean is not None:
                high = np.asarray(input_mean, np.float32)
                low = np.asarray(input_mean - high, np.float32)
                centering = [session.upload(a) for a in (high, low, np.ones_like(high))]
            wb = [session.upload(np.ascontiguousarray(w, dtype=np.float32)) for w in weights]
            bb = [
                session.upload(np.broadcast_to(b, (w.shape[1],)).astype(np.float32))
                for w, b in zip(weights, biases, strict=True)
            ]
            # Reuse the activation buffers between chunks within this call.
            buffers = [
                session.empty((min(rows, len(x)), width))
                for width in [x.shape[1]] + [w.shape[1] for w in weights]
            ]
            for start in range(0, len(x), rows):
                count = min(rows, len(x) - start)
                buffers[0].array[:count] = x[start : start + count]
                self.uploaded_bytes += count * x.shape[1] * 4
                command = session.command()
                dispatches = 0
                if centering is not None:
                    session.encode(
                        command,
                        "standardize",
                        [buffers[0], *centering, buffers[0]],
                        [count, x.shape[1]],
                        (count * x.shape[1],),
                    )
                    dispatches += 1
                for i, (w, act) in enumerate(zip(weights, activations, strict=True)):
                    left, result = buffers[i : i + 2]
                    n, d, c = count, w.shape[0], w.shape[1]
                    if c <= 8:
                        session.encode(
                            command,
                            "linear_small",
                            [left, wb[i], bb[i], result],
                            [n, d, c, codes[act]],
                            (n * c,),
                        )
                        dispatches += 1
                    else:
                        if self.mps is not None:
                            session.matmul(command, left, wb[i], result)
                        else:
                            session.encode(
                                command, "matmul", [left, wb[i], result], [n, c, d], (c, n)
                            )
                        session.encode(
                            command, "activation", [result, bb[i]], [n, c, codes[act]], (n * c,)
                        )
                        dispatches += 2
                    if act == "softmax":
                        session.encode(command, "softmax_rows", [result], [n, c], (n,))
                        dispatches += 1
                session.submit(command, dispatches=dispatches)
                output[start : start + count] = buffers[-1].array[:count]
        return output

    def gaussian_scores(self, x, mean, variance, prior):
        bias = np.log(prior) - 0.5 * np.log(2 * np.pi * variance).sum(axis=1)
        return self.run(
            "gaussian_scores",
            [
                x,
                np.asarray(mean, np.float32),
                np.asarray(variance, np.float32),
                np.asarray(bias, np.float32),
            ],
            [((len(x), len(mean)), np.float32)],
            [len(x), x.shape[1], len(mean)],
            (len(x) * len(mean),),
        )[0]

    def tree_scores(self, x, trees, classifier):
        # Repack on every call: user edits and warm_start cannot leave stale state.
        sizes = [t.node_count for t in trees]
        offsets = np.r_[0, np.cumsum(sizes)[:-1]].astype(np.int32)
        left = np.concatenate(
            [t.children_left + o for t, o in zip(trees, offsets, strict=True)]
        ).astype(np.int32)
        right = np.concatenate(
            [t.children_right + o for t, o in zip(trees, offsets, strict=True)]
        ).astype(np.int32)
        features = np.concatenate([t.feature for t in trees]).astype(np.int32)
        threshold64 = np.concatenate([t.threshold for t in trees])
        thresholds = threshold64.astype(np.float32)
        # X is float32: round thresholds downward so branch decisions match
        # sklearn's comparisons against double-precision split thresholds exactly.
        rounded_up = thresholds.astype(np.float64) > threshold64
        thresholds[rounded_up] = np.nextafter(thresholds[rounded_up], -np.inf)
        values = np.concatenate(
            [t.value[:, 0, :] if classifier else t.value[:, :, 0] for t in trees]
        ).astype(np.float32)
        rows = max(1, (32 * 1024 * 1024) // (4 * len(trees) * values.shape[1]))
        result = np.empty((len(x), values.shape[1]), np.float32)
        with Session(self) as session:
            state = [
                session.upload(a) for a in (offsets, left, right, features, thresholds, values)
            ]
            xb = session.empty((min(rows, len(x)), x.shape[1]))
            partial = session.empty((len(trees), min(rows, len(x)), values.shape[1]))
            out = session.empty((min(rows, len(x)), values.shape[1]))
            device_state, device_x = None, None
            for start in range(0, len(x), rows):
                count = min(rows, len(x) - start)
                xb.array[:count] = x[start : start + count]
                self.uploaded_bytes += count * x.shape[1] * 4
                command = session.command()
                if device_state is None:
                    device_state = [session.private_copy(command, b) for b in state]
                device_x = session.private_copy(command, xb, device_x)
                params = [count, x.shape[1], len(trees), values.shape[1]]
                session.encode(
                    command,
                    "tree_scores",
                    [device_x, *device_state, partial],
                    params,
                    (count * len(trees),),
                )
                session.encode(
                    command, "forest_average", [partial, out], params, (count * values.shape[1],)
                )
                session.submit(command, dispatches=2)
                result[start : start + count] = out.array[:count]
        return result

    def svm_scores(self, x, support, coefficients, intercept, kernel, gamma, coef0, degree):
        # Limit the kernel matrix to 32 MiB and upload support vectors once.
        rows = max(1, (32 * 1024 * 1024) // (4 * len(support)))
        output = np.empty((len(x), coefficients.shape[1]), np.float32)
        with Session(self) as session:
            sv = session.upload(np.asarray(support, np.float32))
            coef = session.upload(np.ascontiguousarray(coefficients, dtype=np.float32))
            bias = session.upload(np.asarray(intercept, np.float32))
            kp = session.upload(np.array([gamma, coef0, degree], np.float32))
            xb = session.empty((min(rows, len(x)), x.shape[1]))
            kb = session.private_empty(min(rows, len(x)) * len(support) * 4)
            result = session.empty((min(rows, len(x)), coefficients.shape[1]))
            device_sv, device_x = None, None
            for start in range(0, len(x), rows):
                n = min(rows, len(x) - start)
                xb.array[:n] = x[start : start + n]
                self.uploaded_bytes += n * x.shape[1] * 4
                command = session.command()
                if device_sv is None:
                    device_sv = session.private_copy(command, sv)
                device_x = session.private_copy(command, xb, device_x)
                session.encode(
                    command,
                    "svm_kernel",
                    [device_x, device_sv, kp, kb],
                    [
                        n,
                        x.shape[1],
                        len(support),
                        {"linear": 0, "rbf": 1, "poly": 2, "sigmoid": 3}[kernel],
                    ],
                    (n * len(support),),
                )
                session.encode(
                    command,
                    "linear_small",
                    [kb, coef, bias, result],
                    [n, len(support), coefficients.shape[1], 0],
                    (n * coefficients.shape[1],),
                )
                session.submit(command, dispatches=2)
                output[start : start + n] = result.array[:n]
        return output

    def neighbors(self, x, training, k):
        rows = max(1, (32 * 1024 * 1024) // (4 * len(training)))
        distance = np.empty((len(x), k), np.float32)
        index = np.empty((len(x), k), np.intp)
        with Session(self) as session:
            train = session.upload(training)
            xb = session.empty((min(rows, len(x)), x.shape[1]))
            matrix = session.private_empty(min(rows, len(x)) * len(training) * 4)
            db = session.empty((min(rows, len(x)), k))
            ib = session.empty((min(rows, len(x)), k), np.int32)
            device_train, device_x = None, None
            for start in range(0, len(x), rows):
                n = min(rows, len(x) - start)
                xb.array[:n] = x[start : start + n]
                self.uploaded_bytes += n * x.shape[1] * 4
                command = session.command()
                if device_train is None:
                    device_train = session.private_copy(command, train)
                device_x = session.private_copy(command, xb, device_x)
                session.encode(
                    command,
                    "distances_tiled",
                    [device_x, device_train, matrix],
                    [n, len(training), x.shape[1]],
                    (len(training), n),
                )
                session.encode(command, "knn_topk", [matrix, db, ib], [n, len(training), k], (n,))
                session.submit(command, dispatches=2)
                distance[start : start + n] = db.array[:n]
                index[start : start + n] = ib.array[:n]
        return distance, index
