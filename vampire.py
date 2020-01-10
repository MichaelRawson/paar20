import clauses
import subprocess

VAMPIRE_STARTUP = 42.7E6

class VampireCrashed(Exception):
    pass

def clausify(path, timeout=1.0):
    clausify = subprocess.Popen([
        'vampire',
        '--mode', 'clausify',
        path,
    ], stdout=subprocess.PIPE)
    clause_ascii, _ = clausify.communicate(timeout=timeout)
    return clauses.parse(clause_ascii)

def score(clauses, timeout=1.0):
    proc = subprocess.Popen([
        'perf', 'stat',
        '-e', 'instructions:u',
        '-x', ',',
        'vampire',
        '-p', 'off',
        '-av', 'off',
        '-sa',  'discount'
    ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    _, stderr = proc.communicate(input=b''.join(clauses), timeout=timeout)

    try:
        return int(stderr.split(b',')[0]) - VAMPIRE_STARTUP
    except ValueError:
        raise VampireCrashed()

def strip_prefix(clause):
    return clause.split(b',', 2)[2]

def infer(existing):
    existing_formulae = set(strip_prefix(clause) for clause in existing)
    output = subprocess.check_output([
        'vampire',
        '-av', 'off',
        '-sa',  'discount',
        '-awr', '1:0',
        '--max_age', '1'
    ], input=b''.join(existing))
    if b'% SZS status Unsatisfiable' in output:
        return None

    saturated = clauses.parse(output)
    return [
        sat for sat in saturated if
        strip_prefix(sat) not in existing_formulae
    ]
