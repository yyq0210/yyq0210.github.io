"""CPU-only validation of indexing; not a GPU execution or timing test."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
BM,BN,BK,MMAK=128,256,256,16
checks=[]
def record(name, detail):checks.append({'check':name,'result':'PASS','detail':detail})

def smem_offset(r,q,h):return (q//8)*h*16+r*16+(q%8)*2

for h in (BM,BN):
    offsets=[smem_offset(r,q,h) for r in range(h) for q in range(BK)]
    assert len(set(offsets))==h*BK
    assert sorted(offsets)==list(range(0,h*BK*2,2))
record('SMEM bijection','A 32768 and B 65536 BF16 slots are covered exactly once; all offsets even and in bounds.')

M=N=K=4096
bid=35
grid_n=N//BN
bm,bn=divmod(bid,grid_n)
offm,offn=bm*BM,bn*BN
t=1
for h,offrow in ((BM,offm),(BN,offn)):
    for j in range(BK//8):
        for r in range(h):
            for u in range(8):
                q=j*8+u
                src2=2*((offrow+r)*K+t*BK+q)
                x,y,z=u,offrow+r,t*(BK//8)+j
                src3=x*2+y*K*2+z*16
                dst2=j*h*16+r*16+u*2
                dst3=2*(j*h*8+r*8+u)
                assert src2==src3 and dst2==dst3==smem_offset(r,q,h)
record('2D / 3D TMA equivalence','98304 transferred element coordinates have identical source and shared destination addresses.')

for h in (BM,BN):
    for i in range(BK//MMAK):
        start=i*h*32
        for r in range(h):
            for v in range(MMAK):
                local=start+(v//8)*(h*16)+(r//8)*128+(r%8)*16+(v%8)*2
                assert local==smem_offset(r,i*MMAK+v,h)
record('MMA descriptor reconstruction','MMA start + LBO + SBO + row/element offsets matches SMEM layout for every input element.')

seen=set()
for tid in range(128):
    warp,lane=divmod(tid,32)
    for n in range(BN//8):
        addr=(warp*32<<16)+n*8
        for i in range(8):
            r,c=(addr>>16)+lane,(addr&0xffff)+i
            assert r==tid and c==n*8+i
            assert (r,c) not in seen
            seen.add((r,c))
assert len(seen)==BM*BN
record('TMEM / epilogue coverage','4 warps x 32 lanes x 32 iterations x 8 values cover all 32768 output elements without overlap.')

for b in range((M//BM)*(N//BN)):
    r,c=divmod(b,N//BN)
    assert 0<=r<M//BM and 0<=c<N//BN
assert (bm,bn,offm,offn)==(2,3,256,768)
assert smem_offset(35,17,BM)==4658
assert smem_offset(18,17,BN)==8482
assert ((35<<16)+18)==2293778
assert (offm+35,offn+2*8+2)==(291,786)
assert smem_offset(9,25,BM)==6290
assert 7*BM*32==28672 and 7*BN*32==57344
record('Worked examples and grid','Verified trace C[291,786], bid35, all exercise answers, and 512 CTA coordinates.')

assert (BM+BN)*BK*2==196608
assert BM*BN*4==131072
assert BM*BN*2==65536
assert (BK//MMAK)*(K//BK)==256
assert 2*BM*BN*K*((M//BM)*(N//BN))==2*M*N*K
record('Byte counts / FLOP counts','192 KiB input SMEM; 128 KiB FP32 accumulator; 64 KiB output; total work equals 2MNK.')

report={'scope':'CPU index arithmetic only; no GPU compilation, synchronization test, numerical GEMM test, or benchmark', 'checks':checks}
(ROOT/'layout-check-results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))
