#!/usr/bin/env python3
"""vmad_alias_scan.py - look for the sm_60 STRATA_DP4A register-aliasing pattern (#4274) in real SASS.

    python3 -I vmad_alias_scan.py <object or binary> [...]

strata_dp4a(a, b, c) on sm_60 is four chained `VMAD.S8.S8 Rd, Ra.Bk, Rb.Bk, Racc`. Correct code reads a and b
(the byte-selected operands) from registers that a load or ALU op wrote; the accumulator chains through Rd.
The aliasing bug gives the output the register of a or b, so a later VMAD of the chain reads `Rd.Bk` that an
earlier VMAD just wrote. The exact sign: a VMAD whose a or b operand selects byte 1, 2 or 3 of the SAME register
as its own accumulator (operand 4), i.e. it reads input bytes out of the running sum. Correct code never does
that (byte 0 of the chain may legitimately share it only for strata_dp4a(a, b, a), which then breaks at byte 1).
"""
import re
import subprocess
import sys

CUOBJDUMP = "/usr/local/cuda-12.8/bin/cuobjdump"
FUNC = re.compile(r"^\s*Function\s*:\s*(\S+)")
INSN = re.compile(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?P\d\s+)?([A-Z0-9_.]+)\s+([^;]*);")
REG = re.compile(r"^-?\|?(R\d+)")


def regs_of(op: str):
    m = REG.match(op.strip())
    return m.group(1) if m else None


def scan(path: str):
    out = subprocess.run([CUOBJDUMP, "-sass", path], capture_output=True, text=True).stdout
    func, hits, n_vmad = "?", [], 0
    for line in out.splitlines():
        f = FUNC.match(line)
        if f:
            func = f.group(1)
            continue
        m = INSN.search(line)
        if not m or not m.group(1).startswith("VMAD"):
            continue
        n_vmad += 1
        args = [a.strip() for a in m.group(2).split(",")]
        if len(args) < 4:
            continue
        acc = regs_of(args[3])
        for src in args[1:3]:
            if re.search(r"\.B[123]\b", src) and regs_of(src) == acc and acc is not None:
                hits.append((func, line.strip()))
                break
    return n_vmad, hits


def main():
    total = 0
    for p in sys.argv[1:]:
        n, hits = scan(p)
        total += len(hits)
        print(f"{p}: {n} VMAD, {len(hits)} aliased-input VMAD")
        for func, line in hits[:5]:
            print(f"    {func}: {line}")
    print(f"total aliased-input VMAD: {total}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
