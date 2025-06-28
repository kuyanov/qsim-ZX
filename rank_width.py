import subprocess
import sys


def rank_decomposition(g, timeout_s=1):
    if g.num_vertices() == 0:
        return 0, []
    vs = {v: i for i, v in enumerate(g.vertices())}
    es = [(vs[u], vs[v]) for u, v in g.edges()]
    filename = 'graph.dgf'
    graph_text = f'p edge {len(vs)} {len(es)}\n' + '\n'.join([f'{u} {v}' for u, v in es])
    with open(filename, 'w') as f:
        f.write(graph_text)
    binary = "rank-width-build/RankWidthApproximate"
    cmd = [binary, '-tl', str(timeout_s), filename]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        tree_edges = []
        for line in result.stdout.splitlines():
            if line.startswith('node') or line.startswith('leaf'):
                tokens = line.split()
                u = int(tokens[0].split('_')[1])
                v = int(tokens[2].split('_')[1])
                r = int(tokens[3][1:-1].split('=')[1])
                tree_edges.append((u, v, r))
        if tree_edges:
            rw = max([r for u, v, r in tree_edges])
            return rw, tree_edges

        print("RankWidthApproximate not terminated normally:\n" + result.stderr, file=sys.stderr)
        print("Input:", file=sys.stderr)
        print(graph_text, end='\n\n', file=sys.stderr)
        return -1, None
    except subprocess.CalledProcessError as e:
        print("Error running RankApproximate:", e.stderr, file=sys.stderr)
        return -1, None
