"""CPU-only verification of tutorial address formulas; no GPU dependencies."""
import json
from pathlib import Path

def v1(m,k,h):
    return (k//8)*h*16+m*16+(k%8)*2

def v2(m,k,h):
    return (k//64)*h*128+m*128+(((k%64)//8)^(m%8))*16+(k%8)*2

def v2_bits(m,k,h):
    raw=(k//64)*h*128+m*128+(k%64)*2
    return raw^((raw>>3)&0x70)

def main():
    report={'gpu_execution_performed':False,'basis':'BF16; K tile 256; 1024B-aligned bases; v2 128B swizzle with 16B atomicity','matrices':[]}
    for h in (128,256):
        count=h*256
        for name,fn in [('v1',v1),('v2',v2)]:
            offsets=[fn(m,k,h) for m in range(h) for k in range(256)]
            assert set(offsets)==set(range(0,count*2,2))
            physical=[None]*count
            for logical,off in enumerate(offsets):physical[off//2]=logical
            assert [physical[off//2] for off in offsets]==list(range(count))
            report['matrices'].append({'height':h,'layout':name,'elements':count,'unique_offsets':len(set(offsets)),'min_byte_offset':min(offsets),'max_byte_offset':max(offsets),'round_trip':True})
        assert all(v2(m,k,h)==v2_bits(m,k,h) for m in range(h) for k in range(256))
        # Whole 8-row groups and corresponding half-K packs.
        assert all(v1(m+8,k,h)-v1(m,k,h)==128 for m in range(h-8) for k in range(256))
        assert all(v2(m+8,k,h)-v2(m,k,h)==1024 for m in range(h-8) for k in range(256))
        assert all(v1(m,k+8,h)-v1(m,k,h)==h*16 for m in range(h) for k in range(0,256,16))
        for j in range(16):
            assert v1(0,j*16,h)==j*h*32
            assert v2(0,j*16,h)==(j//4)*h*128+(j%4)*32
        original=[(0,0)]+[(0,k2) for k2 in range(1,4)]+[(k1,k2) for k1 in range(1,4) for k2 in range(4)]
        flat=[(j//4,j%4) for j in range(16)]
        assert original==flat
        for iter_k in range(16):
            original_acc=[iter_k!=0]+[True]*15
            flat_acc=[iter_k!=0 or j!=0 for j in range(16)]
            assert original_acc==flat_acc
        coverage=[i*256+k1*64+k2*16+inner for i in range(16) for k1,k2 in flat for inner in range(16)]
        assert coverage==list(range(4096))
    examples=[(0,0,0,0),(1,0,16,144),(0,8,2048,16),(1,10,2068,132),(1,74,18452,16516)]
    for m,k,a,b in examples:
        assert (v1(m,k,128),v2(m,k,128))==(a,b)
    flops=2*4096**3
    names=['cublas','v1a','v1b','v2a','v2b'];times=[.082,.523,.521,.184,.180];rates=[1673,263,264,748,764]
    for ms,tflops in zip(times,rates):assert abs(flops/(ms*1e-3)/1e12-tflops)/tflops<.01
    report.update({'byte_swizzle_matches_pack_swizzle':True,'lbo_sbo_distances_match':True,'mma_start_matches_element_formula':True,'single_loop_matches_original_sequence':True,'accumulation_predicates_match':True,'full_global_k_coverage':4096,'numeric_examples':examples,'benchmark_flops':flops,'benchmark_rounding_consistent_within_1_percent':True,'v2a_over_v1a':748/263,'v2b_over_v1b':764/264,'v2b_over_cublas':764/1673,'v2b_over_v2a':764/748})
    target=Path(__file__).with_name('verification.json')
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
