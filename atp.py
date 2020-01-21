import time
import subprocess

import clauses

VAMPIRE_STARTUP = 4e7

class Crashed(Exception):
    pass

class ProvedIt(Exception):
    pass

class Timeout(Exception):
    pass

def clausify(path, timeout=1.0):
    clausify = subprocess.Popen([
        'vampire',
        '-t', '1',
        '--mode', 'clausify',
        path,
    ], stdout=subprocess.PIPE)
    try:
        clause_ascii, _ = clausify.communicate(timeout=timeout)
    except subprocess.CalledProcessError:
        raise Timeout()
    except subprocess.TimeoutExpired:
        raise Timeout()

    return clauses.parse(clause_ascii)

def score(clauses, timeout=1.0):
    try:
        proc = subprocess.Popen([
            'perf', 'stat',
            '-e', 'instructions:u',
            '-x', ',',
            'vampire',
            '-p', 'off',
            '-av', 'off',
            '-sa',  'discount'
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        # subprocess communicate() not reliable for grandchild procs
        proc.stdin.write(b''.join(clauses))
        proc.stdin.close()
        proc.wait(timeout)
        stderr = proc.stderr.read()
    except subprocess.CalledProcessError:
        raise Crashed()
    except subprocess.TimeoutExpired:
        proc.terminate()
        proc.kill()
        raise Timeout()

    try:
        return int(stderr.split(b',')[0]) - VAMPIRE_STARTUP
    except ValueError:
        raise Crashed()

def strip_prefix(clause):
    return clause.split(b',', 2)[2]

def infer(existing):
    existing_formulae = set(strip_prefix(clause) for clause in existing)
    try:
        output = subprocess.check_output([
            'vampire',
            '-t', '1',
            '-av', 'off',
            '-sa',  'discount',
            '-awr', '1:0',
            '--max_age', '1'
        ], input=b''.join(existing))
    except subprocess.CalledProcessError:
        raise Timeout()
    except subprocess.TimeoutExpired:
        raise Timeout()

    if b'% SZS status Unsatisfiable' in output:
        raise ProvedIt()

    saturated = clauses.parse(output)
    return [
        sat for sat in saturated if
        strip_prefix(sat) not in existing_formulae
    ]
