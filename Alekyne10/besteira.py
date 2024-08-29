import random
import matplotlib.pyplot as plt
import networkx as nx

# Definindo as expressões
expr1 = "((7 + 3) * (5 - 2)) / ((4 - 3) * (3 - 1))"
expr2 = "(A + ((B - C)) * (D % (E * F)))"


# Função para construir a árvore binária a partir de uma expressão
def build_tree(expr):
    stack = []
    for char in expr:
        if char.isalnum() or char in ['+', '-', '*', '/', '%']:
            node = Node(char)
            stack.append(node)
        elif char == ')':
            right = stack.pop()
            operator = stack.pop()
            left = stack.pop()
            operator.left = left
            operator.right = right
            stack.append(operator)
    return stack[0]


class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


# Criando as duas árvores
tree1 = build_tree(expr1)
tree2 = build_tree(expr2)


# Função para escolher um nó aleatório em uma árvore
def random_node(tree):
    nodes = []

    def traverse(node):
        if node:
            nodes.append(node)
            traverse(node.left)
            traverse(node.right)

    traverse(tree)
    return random.choice(nodes)


# Função de crossover
def crossover(node1, node2):
    temp = node1.left
    node1.left = node2.left
    node2.left = temp


# Selecionando nós aleatórios para crossover
node1 = random_node(tree1)
node2 = random_node(tree2)

# Realizando o crossover
crossover(node1, node2)


# Função para desenhar a árvore
def draw_tree(node, pos=None, graph=None, level=0, width=2.):
    if pos is None:
        pos = {node: (0, 0)}
    if graph is None:
        graph = nx.Graph()

    width = width / 2

    if node.left:
        pos[node.left] = (pos[node][0] - width, pos[node][1] - 1)
        graph.add_edge(node, node.left)
        pos, graph = draw_tree(node.left, pos=pos, graph=graph, level=level + 1, width=width)

    if node.right:
        pos[node.right] = (pos[node][0] + width, pos[node][1] - 1)
        graph.add_edge(node, node.right)
        pos, graph = draw_tree(node.right, pos=pos, graph=graph, level=level + 1, width=width)

    return pos, graph


# Função para plotar a árvore
def plot_tree(tree, title):
    pos, graph = draw_tree(tree)
    labels = {node: node.value for node in pos}
    nx.draw(graph, pos, labels=labels, with_labels=True, node_size=3000, node_color="skyblue", font_size=15)
    plt.title(title)
    plt.show()


# Plotando as árvores
plot_tree(tree1, "Árvore 1 (Pai)")
plot_tree(tree2, "Árvore 2 (Pai)")

# Plotando os filhos após o crossover
plot_tree(tree1, "Árvore 1 (Filho)")
plot_tree(tree2, "Árvore 2 (Filho)")
