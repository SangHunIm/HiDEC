from anytree import Node, RenderTree
from pathlib import Path

root = "root"
def get_tree(node_file, tree_file):
    d = {}
    v = {"[pad]":0,"(":1, ")":2, "[END]":3}
    lines = Path(node_file).open().readlines()
    idx = 4
    for line in lines:
        id, disc = line.strip().split("\t")
        #id = line.strip()
        d[id] = Node(id)
        v[id] = idx
        idx += 1

    lines = Path(tree_file).open().readlines()

    for line in lines:
        p, c = line.strip().split("\t")

        d[c].parent = d[p]

    return d, v

def tree2vector(tree):
    text = []
    if tree.is_leaf:
        text += ["(",tree.name,")"]
        return text
    else:
        text += ["(", tree.name]
        for child in tree.children:
            tmp = tree2vector(child)
            text+=tmp
        return text + [")"]    

def nodes2vector(nodes, tree, vocab):
    tmp_tree = {}
    nodes = sorted(nodes, key=lambda x: tree[x].depth)
    for node in nodes:
        for n in tree[node].ancestors:
            if n.name in tmp_tree:
                continue
            if n.is_root:
                tmp_tree[n.name] = Node(n.name)
            else:
                name = n.name
                parent = n.parent.name
                tmp_tree[name] = Node(name, parent=tmp_tree[parent])
        if node not in tmp_tree:
            tmp_tree[node] = Node(node, parent=tmp_tree[tree[node].parent.name])
        tmp_tree[node+"_END"] = Node("[END]", parent=tmp_tree[node])
    input_token = tree2vector(tmp_tree[root])
    input_idx = [vocab[t] for t in input_token]
    level_idx = []
    target_position = []
    target_children = []
    target_golden = []
    tmp = 0
    for idx, token in zip(input_idx, input_token):
        if idx == 1:
            tmp+=1
            level_idx.append(tmp)
            target_position.append(False)
        elif idx == 2:
            level_idx.append(tmp)
            target_position.append(False)
            tmp-=1
        else:
            level_idx.append(tmp)
            if token == "[END]":
                target_position.append(False)
            elif tree[token].is_leaf:
                target_position.append(False)
            else:
                target_position.append(True)
                target_children.append([vocab[n.name] for n in tree[token].children]+[vocab["[END]"]])
                if tmp_tree[token].is_leaf:
                    target_golden.append([vocab["[END]"]])
                else:    
                    target_golden.append([vocab[n.name] for n in tmp_tree[token].children])

    return input_token, input_idx, level_idx, target_position, target_children, target_golden
    

