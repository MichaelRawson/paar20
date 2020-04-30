import subprocess
import clauses

VAMPIRE = 'vampire'
VAMPIRE_STARTUP = 4e7
TIMEOUT = 1.0

class Crashed(Exception):
    pass

class ProvedIt(Exception):
    pass

class Timeout(Exception):
    pass

def tptp_clause(is_conjecture, clause):
    role = "negated_conjecture" if is_conjecture else "axiom"
    return f"cnf(c, {role}, {clause}).\n".encode('ascii')

def clausify(path):
    try:
        tptp_bytes = subprocess.check_output([
            VAMPIRE,
            '--mode', 'clausify',
            path
        ], timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        raise Timeout()

    parsed = clauses.parse(tptp_bytes)
    axioms = []
    conjectures = []
    for is_conjecture, clause in parsed:
        if is_conjecture:
            conjectures.append((is_conjecture, clause))
        else:
            axioms.append((is_conjecture, clause))
    return axioms, conjectures

def score(axioms, selected):
    try:
        proc = subprocess.Popen([
            'perf', 'stat',
            '-e', 'instructions:u',
            '-x', ',',
            VAMPIRE,
            '-t', str(TIMEOUT),
            '-p', 'off',
            '-av', 'off',
            '-sa',  'discount'
        ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
        for is_conjecture, clause in reversed(selected):
            proc.stdin.write(tptp_clause(is_conjecture, clause))
        for is_conjecture, clause in reversed(axioms):
            assert not is_conjecture
            proc.stdin.write(tptp_clause(is_conjecture, clause))
        proc.stdin.close()
        proc.wait(2 * TIMEOUT)
        if proc.returncode != 0:
            raise Crashed()
        stderr = proc.stderr.read()
    except subprocess.TimeoutExpired:
        proc.terminate()
        proc.kill()
        raise Timeout()

    try:
        return int(stderr.split(b',')[0]) - VAMPIRE_STARTUP
    except ValueError:
        raise Crashed()

def infer(selected):
    existing = set(selected)
    try:
        proc = subprocess.Popen([
            VAMPIRE,
            '-av', 'off',
            '-sa',  'discount',
            '--max_age', '1'
        ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        for is_conjecture, clause in reversed(selected):
            proc.stdin.write(tptp_clause(is_conjecture, clause))

        proc.stdin.close()
        proc.wait(2 * TIMEOUT)
        if proc.returncode != 0:
            raise Crashed()
        tptp_bytes = proc.stdout.read()
    except subprocess.TimeoutExpired:
        proc.terminate()
        proc.kill()
        raise Timeout()

    if b"SZS status Unsatisfiable" in tptp_bytes:
        raise ProvedIt()

    parsed = clauses.parse(tptp_bytes)
    inferred = [
        inference for inference in parsed
        if not inference in existing
    ]
    return inferred
