import subprocess
import clauses

VAMPIRE = 'vampire'
VAMPIRE_STARTUP = 4e7
TIMEOUT = 10.0

class Crashed(Exception):
    pass

class ProvedIt(Exception):
    pass

class Timeout(Exception):
    pass

def tptp_clause(role, clause):
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

    axioms, conjectures, extras = clauses.parse(tptp_bytes)
    axioms = [('axiom', axiom) for axiom in axioms]
    conjectures = [('negated_conjecture', conjecture) for conjecture in conjectures]
    extras = [('type', extra) for extra in extras]
    return axioms, conjectures, extras

def score(axioms, selected, extras):
    try:
        proc = subprocess.Popen([
            'perf', 'stat',
            '-e', 'instructions:u',
            '-x', ',',
            VAMPIRE,
            '-t', str(1.1 * TIMEOUT),
            '-p', 'off',
            '-av', 'off',
            '-sa',  'discount',
        ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
        for role, clause in extras:
            proc.stdin.write(tptp_clause(role, clause))
        for role, clause in reversed(selected):
            proc.stdin.write(tptp_clause(role, clause))
        for role, clause in reversed(axioms):
            proc.stdin.write(tptp_clause(role, clause))
        proc.stdin.close()
        proc.wait(TIMEOUT)
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

def infer(selected, extras):
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
            #stderr=subprocess.DEVNULL
        )
        for role, clause in extras:
            proc.stdin.write(tptp_clause(role, clause))
        for role, clause in reversed(selected):
            proc.stdin.write(tptp_clause(role, clause))

        proc.stdin.close()
        proc.wait(1.0)
        if proc.returncode != 0:
            print(extras)
            print(proc.stdout.read())
            raise Crashed()
        tptp_bytes = proc.stdout.read()
    except subprocess.TimeoutExpired:
        proc.terminate()
        proc.kill()
        raise Timeout()

    if b"SZS status Unsatisfiable" in tptp_bytes:
        raise ProvedIt()

    axioms, conjectures, extras = clauses.parse(tptp_bytes)
    axioms = [('axiom', axiom) for axiom in axioms]
    conjectures = [('negated_conjecture', conjecture) for conjecture in conjectures]
    extras = [('type', extra) for extra in extras]

    inferred = [
        inference for inference in axioms + conjectures
        if not inference in existing
    ]
    return inferred, extras
