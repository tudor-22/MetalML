#include <metal_stdlib>
using namespace metal;

// Coalesced feature tiles and independent sample blocks for large reductions.
kernel void stats_partial(device const float* x [[buffer(0)]],
                          device float* partial [[buffer(1)]],
                          constant uint* p [[buffer(2)]],
                          uint2 group [[threadgroup_position_in_grid]],
                          uint2 lane [[thread_position_in_threadgroup]]) {
    uint n=p[0],d=p[1],j=group.x*16+lane.x;
    threadgroup float means[16][16], m2s[16][16], lows[16][16], highs[16][16];
    threadgroup uint counts[16][16];
    float mean=0,m2=0,lo=INFINITY,hi=-INFINITY; uint count=0;
    for(uint i=group.y*256+lane.y;i<min((group.y+1)*256,n);i+=16) {
        float raw=j<d ? x[i*d+j] : 0;
        float v=j<d ? raw-x[j] : 0,delta=v-mean;
        count++;mean+=delta/float(count);m2+=delta*(v-mean);lo=min(lo,raw);hi=max(hi,raw);
    }
    means[lane.y][lane.x]=mean;m2s[lane.y][lane.x]=m2;
    lows[lane.y][lane.x]=lo;highs[lane.y][lane.x]=hi;counts[lane.y][lane.x]=count;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint offset=8;offset>0;offset/=2) {
        if(lane.y<offset) {
            uint a=counts[lane.y][lane.x],b=counts[lane.y+offset][lane.x],total=a+b;
            float delta=means[lane.y+offset][lane.x]-means[lane.y][lane.x];
            if(total>0) {
                means[lane.y][lane.x]+=delta*float(b)/float(total);
                m2s[lane.y][lane.x]+=m2s[lane.y+offset][lane.x]+delta*delta*float(a)*float(b)/float(total);
            }
            counts[lane.y][lane.x]=total;
            lows[lane.y][lane.x]=min(lows[lane.y][lane.x],lows[lane.y+offset][lane.x]);
            highs[lane.y][lane.x]=max(highs[lane.y][lane.x],highs[lane.y+offset][lane.x]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(lane.y==0 && j<d) {
        uint base=group.y*4*d;
        partial[base+j]=means[0][lane.x];partial[base+d+j]=m2s[0][lane.x];
        partial[base+2*d+j]=lows[0][lane.x];partial[base+3*d+j]=highs[0][lane.x];
    }
}

kernel void stats_finish(device const float* partial [[buffer(0)]],
                         device float* stats [[buffer(1)]],
                         constant uint* p [[buffer(2)]], uint j [[thread_position_in_grid]]) {
    uint n=p[0],d=p[1],chunks=(n+255)/256; if(j>=d) return;
    float mean=0,m2=0,lo=INFINITY,hi=-INFINITY;uint count=0;
    for(uint c=0;c<chunks;c++) {
        uint base=c*4*d,extra=min(256u,n-c*256),total=count+extra;
        float delta=partial[base+j]-mean;
        mean+=delta*float(extra)/float(total);
        m2+=partial[base+d+j]+delta*delta*float(count)*float(extra)/float(total);
        count=total;lo=min(lo,partial[base+2*d+j]);hi=max(hi,partial[base+3*d+j]);
    }
    stats[j]=mean;stats[d+j]=max(m2/float(n),0.0f);stats[2*d+j]=lo;stats[3*d+j]=hi;
}

// Split the contraction dimension of tall/narrow X'X and X'y. Output tiles
// from different sample blocks run independently instead of leaving SMs idle.
kernel void gram_partial(device const float* x [[buffer(0)]],
                         device const float* y [[buffer(1)]],
                         device float* partial [[buffer(2)]],
                         constant uint* p [[buffer(3)]],
                         uint3 group [[threadgroup_position_in_grid]],
                         uint3 lane [[thread_position_in_threadgroup]]) {
    uint n=p[0], d=p[1], targets=p[2], block=p[3], columns=d+targets;
    uint row=group.y*16+lane.y, col=group.x*16+lane.x;
    threadgroup float a[16][16], b[16][16]; float acc=0;
    for(uint base=group.z*block;base<min((group.z+1)*block,n);base+=16) {
        uint sample=base+lane.y, feature=group.y*16+lane.x;
        a[lane.y][lane.x]=(sample<n && feature<d) ? x[sample*d+feature] : 0;
        float v=0;
        if(sample<n && col<columns) v=col<d ? x[sample*d+col] : y[sample*targets+col-d];
        b[lane.y][lane.x]=v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint q=0;q<16;q++) acc+=a[q][lane.y]*b[q][lane.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(row<d && col<columns) partial[(group.z*d+row)*columns+col]=acc;
}

kernel void gram_finish(device const float* partial [[buffer(0)]],
                        device float* result [[buffer(1)]],
                        constant uint* p [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    uint size=p[1]*(p[1]+p[2]), chunks=(p[0]+p[3]-1)/p[3]; if(id>=size) return;
    float sum=0; for(uint c=0;c<chunks;c++) sum+=partial[c*size+id];
    result[id]=sum;
}

kernel void standardize_fitted(device const float* x [[buffer(0)]],
                               device const float* stats [[buffer(1)]],
                               device float* out [[buffer(2)]],
                               constant uint* p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    uint d=p[1]; if(i>=p[0]*d) return;
    uint j=i%d;
    float v=p[2] ? (x[i]-x[j])-stats[j] : x[i];
    float scale=p[3] ? sqrt(stats[d+j]) : 1.0f;
    out[i]=v/(scale==0 ? 1.0f : scale);
}

// One SIMD group cooperates on a row: contiguous feature reads, native reduction.
kernel void assign_coalesced(device const float* x [[buffer(0)]],
                             device const float* centers [[buffer(1)]],
                             device int* labels [[buffer(2)]],
                             device float* errors [[buffer(3)]],
                             device atomic_uint* status [[buffer(4)]],
                             constant uint* p [[buffer(5)]],
                             uint tid [[thread_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint width [[threads_per_simdgroup]]) {
    uint i=tid/width, n=p[0], k=p[1], d=p[2]; if(i>=n) return;
    float best=INFINITY; int label=0;
    for(uint c=0;c<k;c++) {
        float sum=0;
        for(uint j=lane;j<d;j+=width) { float delta=x[i*d+j]-centers[c*d+j];sum+=delta*delta; }
        sum=simd_sum(sum);
        if(sum<best) {best=sum;label=int(c);}
    }
    if(lane==0) {
        labels[i]=label;errors[i]=best;
        if(!isfinite(best)) atomic_fetch_or_explicit(status,1u,memory_order_relaxed);
    }
}

kernel void distances_coalesced(device const float* x [[buffer(0)]],
                                device const float* centers [[buffer(1)]],
                                device float* out [[buffer(2)]],
                                constant uint* p [[buffer(3)]],
                                uint tid [[thread_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint width [[threads_per_simdgroup]]) {
    uint id=tid/width, n=p[0], k=p[1], d=p[2]; if(id>=n*k) return;
    uint row=id/k, c=id%k; float sum=0;
    for(uint j=lane;j<d;j+=width) {float delta=x[row*d+j]-centers[c*d+j];sum+=delta*delta;}
    sum=simd_sum(sum); if(lane==0) out[id]=sum;
}

// Partition the rows so center updates occupy the GPU even for small k and d.
kernel void centers_partial(device const float* x [[buffer(0)]],
                            device const int* labels [[buffer(1)]],
                            device const float* weights [[buffer(2)]],
                            device float* partial [[buffer(3)]],
                            constant uint* p [[buffer(4)]], uint id [[thread_position_in_grid]]) {
    uint n=p[0], k=p[1], d=p[2], block=p[3], chunks=(n+block-1)/block;
    if(id>=chunks*k*(d+1)) return;
    uint chunk=id/(k*(d+1)), c=(id/(d+1))%k, j=id%(d+1);
    float sum=0;
    for(uint i=chunk*block;i<min((chunk+1)*block,n);i++)
        if(labels[i]==int(c)) sum+=weights[i]*(j==d ? 1.0f : x[i*d+j]);
    partial[id]=sum;
}

kernel void centers_finish(device const float* partial [[buffer(0)]],
                           device const float* old [[buffer(1)]],
                           device float* centers [[buffer(2)]],
                           device atomic_uint* status [[buffer(3)]],
                           constant uint* p [[buffer(4)]], uint id [[thread_position_in_grid]]) {
    uint n=p[0], k=p[1], d=p[2], block=p[3], chunks=(n+block-1)/block; if(id>=k*d) return;
    uint c=id/d,j=id%d;float sum=0,count=0;
    for(uint part=0;part<chunks;part++) {
        sum+=partial[(part*k+c)*(d+1)+j];
        count+=partial[(part*k+c)*(d+1)+d];
    }
    centers[id]=count>0 ? sum/count : old[id];
    if(count==0) atomic_fetch_or_explicit(status,2u,memory_order_relaxed);
}

// Parallel Welford reduction of values relative to the first row. Shifting
// preserves low-order information when a column has a large constant offset.
kernel void column_stats(device const float* x [[buffer(0)]],
                         device float* out [[buffer(1)]],
                         constant uint* p [[buffer(2)]],
                         uint lane [[thread_position_in_threadgroup]],
                         uint j [[threadgroup_position_in_grid]]) {
    uint n=p[0], d=p[1];
    threadgroup float means[256], m2s[256], lows[256], highs[256];
    threadgroup uint counts[256];
    float mean=0, m2=0, lo=INFINITY, hi=-INFINITY;
    uint count=0;
    for(uint i=lane;i<n;i+=256) {
        float raw=x[i*d+j], v=raw-x[j], delta=v-mean;
        count++; mean += delta/float(count); m2 += delta*(v-mean);
        lo=min(lo,raw); hi=max(hi,raw);
    }
    means[lane]=mean; m2s[lane]=m2; lows[lane]=lo; highs[lane]=hi; counts[lane]=count;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint offset=128;offset>0;offset/=2) {
        if(lane<offset) {
            uint a=counts[lane], b=counts[lane+offset], total=a+b;
            float delta=means[lane+offset]-means[lane];
            if(total>0) {
                means[lane] += delta*float(b)/float(total);
                m2s[lane] += m2s[lane+offset]+delta*delta*float(a)*float(b)/float(total);
            }
            counts[lane]=total;
            lows[lane]=min(lows[lane],lows[lane+offset]);
            highs[lane]=max(highs[lane],highs[lane+offset]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(lane!=0) return;
    mean=means[0]; m2=m2s[0]; lo=lows[0]; hi=highs[0];
    out[j]=mean; out[d+j]=max(m2/float(n),0.0f);
    out[2*d+j]=lo; out[3*d+j]=hi;
}

kernel void standardize(device const float* x [[buffer(0)]],
                        device const float* high [[buffer(1)]],
                        device const float* low [[buffer(2)]],
                        device const float* scale [[buffer(3)]],
                        device float* out [[buffer(4)]],
                        constant uint* p [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    if(i>=p[0]*p[1]) return;
    uint j=i%p[1]; out[i]=((x[i]-high[j])-low[j])/scale[j];
}

kernel void affine(device const float* x [[buffer(0)]],
                   device const float* scale [[buffer(1)]],
                   device const float* bias [[buffer(2)]],
                   device float* out [[buffer(3)]],
                   constant uint* p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if(i>=p[0]*p[1]) return;
    out[i]=x[i]*scale[i%p[1]]+bias[i%p[1]];
}

kernel void normalize_rows(device const float* x [[buffer(0)]],
                           device float* out [[buffer(1)]],
                           constant uint* p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    uint n=p[0], d=p[1], mode=p[2]; if(i>=n) return;
    // Scale before reducing to prevent overflow for large finite inputs.
    float largest=0; for(uint j=0;j<d;j++) largest=max(largest,abs(x[i*d+j]));
    if(largest==0) { for(uint j=0;j<d;j++) out[i*d+j]=0; return; }
    float norm=0;
    for(uint j=0;j<d;j++) { float v=abs(x[i*d+j])/largest;
        norm += mode==2 ? v*v : v; }
    norm = mode==0 ? 1.0f : (mode==2 ? sqrt(norm) : norm);
    for(uint j=0;j<d;j++) out[i*d+j]=(x[i*d+j]/largest)/norm;
}

// Tiled matrix product. A and B are contiguous row-major matrices.
kernel void matmul(device const float* a [[buffer(0)]],
                   device const float* b [[buffer(1)]],
                   device float* c [[buffer(2)]],
                   constant uint* p [[buffer(3)]],
                   uint2 g [[thread_position_in_grid]],
                   uint2 t [[thread_position_in_threadgroup]]) {
    uint m=p[0], n=p[1], k=p[2];
    threadgroup float aa[16][16]; threadgroup float bb[16][16];
    float acc=0;
    for(uint base=0;base<k;base+=16) {
        aa[t.y][t.x]=(g.y<m && base+t.x<k) ? a[g.y*k+base+t.x] : 0;
        bb[t.y][t.x]=(base+t.y<k && g.x<n) ? b[(base+t.y)*n+g.x] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint q=0;q<16;q++) acc += aa[t.y][q]*bb[q][t.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(g.y<m && g.x<n) c[g.y*n+g.x]=acc;
}

kernel void distances(device const float* x [[buffer(0)]],
                      device const float* centers [[buffer(1)]],
                      device float* out [[buffer(2)]],
                      constant uint* p [[buffer(3)]], uint id [[thread_position_in_grid]]) {
    uint n=p[0], k=p[1], d=p[2]; if(id>=n*k) return;
    uint row=id/k, col=id%k; float sum=0;
    for(uint j=0;j<d;j++) { float delta=x[row*d+j]-centers[col*d+j]; sum+=delta*delta; }
    out[id]=sum;
}

kernel void assign_clusters(device const float* x [[buffer(0)]],
                            device const float* centers [[buffer(1)]],
                            device int* labels [[buffer(2)]],
                            device float* errors [[buffer(3)]],
                            constant uint* p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    uint n=p[0], k=p[1], d=p[2]; if(i>=n) return;
    float best=INFINITY; int label=0;
    for(uint c=0;c<k;c++) { float sum=0;
        for(uint j=0;j<d;j++) { float delta=x[i*d+j]-centers[c*d+j]; sum+=delta*delta; }
        if(sum<best) { best=sum; label=int(c); }
    }
    labels[i]=label; errors[i]=best;
}

kernel void update_centers(device const float* x [[buffer(0)]],
                           device const int* labels [[buffer(1)]],
                           device const float* weights [[buffer(2)]],
                           device const float* old [[buffer(3)]],
                           device float* centers [[buffer(4)]],
                           constant uint* p [[buffer(5)]], uint id [[thread_position_in_grid]]) {
    uint n=p[0], k=p[1], d=p[2]; if(id>=k*d) return;
    uint c=id/d, j=id%d; float sum=0, count=0;
    for(uint i=0;i<n;i++) if(labels[i]==int(c)) {sum+=weights[i]*x[i*d+j];count+=weights[i];}
    centers[id]=count>0 ? sum/count : old[id];
}

inline float activate_value(float v, uint code) {
    if(code==1) return max(v,0.0f);
    if(code==2) return tanh(v);
    if(code==3) return v>=0 ? 1.0f/(1.0f+exp(-v)) : exp(v)/(1.0f+exp(v));
    if(code==5) return exp(v);
    return v;
}

kernel void linear_small(device const float* x [[buffer(0)]],
                         device const float* w [[buffer(1)]],
                         device const float* bias [[buffer(2)]],
                         device float* out [[buffer(3)]],
                         constant uint* p [[buffer(4)]],
                         uint id [[thread_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint width [[threads_per_simdgroup]]) {
    uint item=id/width,n=p[0],d=p[1],c=p[2]; if(item>=n*c) return;
    uint row=item/c,col=item%c; float total=0;
    for(uint j=lane;j<d;j+=width) total+=x[row*d+j]*w[j*c+col];
    total=simd_sum(total);
    if(lane==0) out[item]=activate_value(total+bias[col],p[3]);
}

kernel void activation(device float* x [[buffer(0)]], device const float* bias [[buffer(1)]],
                       constant uint* p [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if(id<p[0]*p[1]) x[id]=activate_value(x[id]+bias[id%p[1]],p[2]);
}

kernel void softmax_rows(device float* x [[buffer(0)]], constant uint* p [[buffer(1)]],
                         uint row [[thread_position_in_grid]]) {
    if(row>=p[0]) return;uint d=p[1];float peak=-INFINITY,total=0;
    for(uint j=0;j<d;j++) peak=max(peak,x[row*d+j]);
    for(uint j=0;j<d;j++) {float v=exp(x[row*d+j]-peak);x[row*d+j]=v;total+=v;}
    for(uint j=0;j<d;j++) x[row*d+j]/=total;
}

kernel void gaussian_scores(device const float* x [[buffer(0)]],
                            device const float* mean [[buffer(1)]],
                            device const float* variance [[buffer(2)]],
                            device const float* bias [[buffer(3)]],
                            device float* out [[buffer(4)]], constant uint* p [[buffer(5)]],
                            uint id [[thread_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint width [[threads_per_simdgroup]]) {
    uint item=id/width,n=p[0],d=p[1],c=p[2];if(item>=n*c) return;
    uint row=item/c,cls=item%c;float total=0;
    for(uint j=lane;j<d;j+=width) {float v=x[row*d+j]-mean[cls*d+j];total+=v*v/variance[cls*d+j];}
    total=simd_sum(total);if(lane==0) out[item]=bias[cls]-0.5f*total;
}

kernel void tree_scores(device const float* x [[buffer(0)]],
                        device const int* roots [[buffer(1)]],
                        device const int* left [[buffer(2)]],device const int* right [[buffer(3)]],
                        device const int* feature [[buffer(4)]],device const float* threshold [[buffer(5)]],
                        device const float* values [[buffer(6)]],device float* out [[buffer(7)]],
                        constant uint* p [[buffer(8)]],uint row [[thread_position_in_grid]]) {
    uint n=p[0],d=p[1],trees=p[2],c=p[3];if(row>=n*trees) return;
    uint sample=row%n,t=row/n;int node=roots[t];
    while(feature[node]>=0) node=x[sample*d+feature[node]]<=threshold[node] ? left[node] : right[node];
    for(uint j=0;j<c;j++) out[row*c+j]=values[node*c+j];
}

kernel void forest_average(device const float* partial [[buffer(0)]],device float* out [[buffer(1)]],
                           constant uint* p [[buffer(2)]],uint id [[thread_position_in_grid]]) {
    uint n=p[0],trees=p[2],c=p[3];if(id>=n*c) return;
    float total=0;for(uint t=0;t<trees;t++) total+=partial[t*n*c+id];
    out[id]=total/float(trees);
}

kernel void svm_kernel(device const float* x [[buffer(0)]],device const float* sv [[buffer(1)]],
                       device const float* kp [[buffer(2)]],device float* out [[buffer(3)]],
                       constant uint* p [[buffer(4)]],uint id [[thread_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]],uint width [[threads_per_simdgroup]]) {
    uint item=id/width,n=p[0],d=p[1],s=p[2],kind=p[3];if(item>=n*s) return;
    uint row=item/s,col=item%s;float total=0;
    for(uint j=lane;j<d;j+=width) {
        float a=x[row*d+j],b=sv[col*d+j];total+=kind==1 ? (a-b)*(a-b) : a*b;
    }
    total=simd_sum(total);
    if(lane==0) {
        if(kind==1) total=exp(-kp[0]*total);
        if(kind==2) total=kp[2]==0 ? 1.0f : powr(abs(kp[0]*total+kp[1]),kp[2]) * ((kp[0]*total+kp[1]<0 && (uint(kp[2])%2)) ? -1.0f : 1.0f);
        if(kind==3) total=tanh(kp[0]*total+kp[1]);
        out[item]=total;
    }
}

kernel void knn_topk(device const float* matrix [[buffer(0)]],device float* distances [[buffer(1)]],
                     device int* indices [[buffer(2)]],constant uint* p [[buffer(3)]],
                     uint id [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]],
                     uint width [[threads_per_simdgroup]]) {
    uint row=id/width,n=p[0],s=p[1],k=p[2];if(row>=n) return;
    float previous=-INFINITY;int previous_id=-1;
    for(uint rank=0;rank<k;rank++) {
        float best=INFINITY;int best_id=0x7fffffff;
        for(uint sample=lane;sample<s;sample+=width) {
            float v=matrix[row*s+sample];
            if((v>previous || (v==previous && int(sample)>previous_id)) &&
               (v<best || (v==best && int(sample)<best_id))) {best=v;best_id=int(sample);}
        }
        float minimum=simd_min(best);
        int selected=simd_min(best==minimum ? best_id : 0x7fffffff);
        if(lane==0) {distances[row*k+rank]=sqrt(minimum);indices[row*k+rank]=selected;}
        previous=minimum;previous_id=selected;
    }
}

// Reuse feature tiles across 256 exact squared-distance calculations.
kernel void distances_tiled(device const float* x [[buffer(0)]],device const float* y [[buffer(1)]],
                             device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],
                             uint2 group [[threadgroup_position_in_grid]],
                             uint2 lane [[thread_position_in_threadgroup]]) {
    uint n=p[0],s=p[1],d=p[2],row=group.y*16+lane.y,col=group.x*16+lane.x;
    threadgroup float a[16][16],b[16][16];float total=0;
    for(uint base=0;base<d;base+=16) {
        uint feature=base+lane.x,train=group.x*16+lane.y;
        a[lane.y][lane.x]=(row<n && feature<d) ? x[row*d+feature] : 0;
        b[lane.y][lane.x]=(train<s && feature<d) ? y[train*d+feature] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint j=0;j<16;j++) {float delta=a[lane.y][j]-b[lane.x][j];total+=delta*delta;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(row<n && col<s) out[row*s+col]=total;
}

// Center while loading contraction tiles; no full centered matrix is written.
kernel void gram_centered(device const float* x [[buffer(0)]],device const float* y [[buffer(1)]],
                           device const float* means [[buffer(2)]],device float* partial [[buffer(3)]],
                           constant uint* p [[buffer(4)]],uint3 group [[threadgroup_position_in_grid]],
                           uint3 lane [[thread_position_in_threadgroup]]) {
    uint n=p[0],d=p[1],t=p[2],block=p[3],columns=d+t;
    if(group.x>group.y && (group.x+1)*16<=d) return;
    uint row=group.y*16+lane.y,col=group.x*16+lane.x;
    threadgroup float a[16][16],b[16][16];float acc=0;
    for(uint base=group.z*block;base<min((group.z+1)*block,n);base+=16) {
        uint sample=base+lane.y,feature=group.y*16+lane.x;
        float av=0,bv=0;
        if(sample<n && feature<d) {
            av=x[sample*d+feature];if(p[4]) av=(av-means[feature])-means[d+feature];
        }
        if(sample<n && col<columns) {
            if(col<d) {bv=x[sample*d+col];if(p[4]) bv=(bv-means[col])-means[d+col];}
            else {uint j=col-d;bv=y[sample*t+j];if(p[5]) bv=(bv-means[2*d+j])-means[2*d+t+j];}
        }
        a[lane.y][lane.x]=av;b[lane.y][lane.x]=bv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint j=0;j<16;j++) acc+=a[j][lane.y]*b[j][lane.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(row<d && col<columns) partial[(group.z*d+row)*columns+col]=acc;
}

kernel void gram_symmetric_finish(device const float* partial [[buffer(0)]],device float* out [[buffer(1)]],
                                  constant uint* p [[buffer(2)]],uint id [[thread_position_in_grid]]) {
    uint d=p[1],cols=d+p[2],size=d*cols,chunks=(p[0]+p[3]-1)/p[3];if(id>=size)return;
    uint row=id/cols,col=id%cols,source=(col<d && col>row) ? col*cols+row : id;
    float sum=0;for(uint c=0;c<chunks;c++)sum+=partial[c*size+source];out[id]=sum;
}

kernel void projection_tiled(device const float* x [[buffer(0)]],device const float* w [[buffer(1)]],
                              device const float* bias [[buffer(2)]],device const float* means [[buffer(3)]],
                              device float* out [[buffer(4)]],constant uint* p [[buffer(5)]],
                              uint2 group [[threadgroup_position_in_grid]],uint2 lane [[thread_position_in_threadgroup]]) {
    uint n=p[0],d=p[1],c=p[2],row=group.y*16+lane.y,col=group.x*16+lane.x;
    threadgroup float a[16][16],b[16][16];float acc=0;
    for(uint base=0;base<d;base+=16) {
        uint j=base+lane.x,q=base+lane.y;float v=0;
        if(row<n && j<d) {v=x[row*d+j];if(p[3]) v=(v-means[j])-means[d+j];}
        a[lane.y][lane.x]=v;
        b[lane.y][lane.x]=(q<d && col<c) ? w[q*c+col] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint j=0;j<16;j++)acc+=a[lane.y][j]*b[j][lane.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(row<n && col<c)out[row*c+col]=acc+bias[col];
}

// Row-wise scaling (out[i,j] = x[i,j] * scale[i]). Used to fold the IRLS
// working weights into X so X'WX becomes an ordinary Gram product.
kernel void scale_rows(device const float* x [[buffer(0)]],
                       device const float* scale [[buffer(1)]],
                       device float* out [[buffer(2)]],
                       constant uint* p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if(i>=p[0]*p[1]) return;
    out[i]=x[i]*scale[i/p[1]];
}

// One pass over the linear predictor produces both IRLS quantities: the
// square root of the working weight p(1-p) and the score residual p-y.
kernel void logistic_deriv(device const float* eta [[buffer(0)]],
                           device const float* y [[buffer(1)]],
                           device float* root [[buffer(2)]],
                           device float* resid [[buffer(3)]],
                           constant uint* p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if(i>=p[0]) return;
    float v=eta[i];
    float s=v>=0 ? 1.0f/(1.0f+exp(-v)) : exp(v)/(1.0f+exp(v));
    float w=max(s*(1.0f-s),1e-12f);
    root[i]=sqrt(w);
    resid[i]=s-y[i];
}
