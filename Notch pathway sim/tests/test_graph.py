from notch.graph import load_edge_list


def test_load_edge_list_infers_and_bounds(tmp_path):
    p = tmp_path / "e.txt"
    p.write_text("0 1\n1 2\n")
    adj = load_edge_list(p)
    assert len(adj) == 3
    assert 1 in adj[0]
    assert 0 in adj[1]
