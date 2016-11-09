from nltk.tree import *
import types
import random

def left(self):
    assert self.height() > 2, "This is a leave node"
    return self[0]
def right(self):
    assert self.height() > 2, "This is a leave node"
    return self[1]
def isLeaf(self):
    return self.height() == 2
def word(self):
    if self.isLeaf:
        return self[0]
    else:
        return None
def ilabel(self):
    l = self.label().split('+')[0]
    return self.p2i.get(l, None)

def add_attribute(p2i):

    Tree.left = property(left)
    Tree.right = property(right)
    Tree.isLeaf = property(isLeaf)
    Tree.word = property(word)
    Tree.p2i = p2i
    Tree.sem = None
    Tree.ilabel = property(ilabel)

def Traverse_setid(node, pid, tid):
    node.tid = pid+tid
    if node.isLeaf:
        return
    Traverse_setid(node.left, node.tid, '0')
    Traverse_setid(node.right, node.tid, '1')

def loadTrees(path, p2i, label):
    nltk_trees = []
    add_attribute(p2i)
    with open(path, 'r') as f:
        data = f.read().strip()
        Ts = data.split('\n\n')
        nltk_trees = [Tree.fromstring(t) for t in Ts]
        for t in nltk_trees:
            Tree.chomsky_normal_form(t)
            t.collapse_unary(collapsePOS = True, collapseRoot=True)
            t.sem = label

    for t in nltk_trees:
        Traverse_setid(t, '', 'r')

    # print nltk_trees[0].tid
    # print nltk_trees[0].left.tid
    # print nltk_trees[0].left.left.word
    # print nltk_trees[0].ilabel, p2i[nltk_trees[0].label()], nltk_trees[0].label()
    # nltk_trees[0].pretty_print()

    return nltk_trees

def get_total_postag(args):

    postag = []
    with open(args.pos_data, 'r') as f:
        for line in f.readlines():
            line = line.strip().split()
            postag += [tag[1:] for tag in line if tag.startswith('(')]
    with open(args.neg_data, 'r') as f:
        for line in f.readlines():
            line = line.strip().split()
            postag += [tag[1:] for tag in line if tag.startswith('(')]
    with open(args.testing_data, 'r') as f:
        for line in f.readlines():
            line = line.strip().split()
            postag += [tag[1:] for tag in line if tag.startswith('(')]
    postag = set(postag)
    postag2i = {}
    i2postag = {}
    for i, tag in enumerate(postag):
        postag2i[tag] = i
        i2postag[i] = tag

    return postag2i, i2postag, len(postag2i)

def load_tree_from_data(args):

    postag2i, i2postag, num_tag = get_total_postag(args) # return a postag2index dictionary
    train_tree = []
    if args.type == 'train':
        pos_tree = loadTrees(args.pos_data, postag2i, 1)
        neg_tree = loadTrees(args.neg_data, postag2i, 0)
        train_tree = pos_tree + neg_tree
        random.shuffle(train_tree)

    test_tree = loadTrees(args.testing_data, postag2i, None)

    return train_tree, test_tree, len(postag2i)
