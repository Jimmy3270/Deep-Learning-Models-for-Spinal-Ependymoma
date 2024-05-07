import graphviz


def plot_detailed_architecture():
    dot = graphviz.Digraph(format='png')

    # 1. Reading data
    dot.node('A', 'Read Data', shape='box')

    # 2. Weak Learner (MLP)
    dot.node('B', 'Weak Learner\n(MLP Classifier)', shape='ellipse')

    # 3. Main Ensemble Model
    dot.node('C1', 'Train MLP', shape='box')
    dot.node('C2', 'Get MLP Output Probabilities', shape='box')
    dot.node('C3', 'Combine Original Features and Probabilities', shape='box')
    dot.node('C4', 'Train LightGBM with Enhanced Features', shape='box')

    # 4. SMOTE
    dot.node('D', 'SMOTE Oversampling', shape='diamond')

    # 5. StratifiedKFold CV
    dot.node('E', 'Stratified K-Fold CV', shape='parallelogram')

    # Setting edges
    dot.edge('A', 'B')
    dot.edge('B', 'C1')
    dot.edge('C1', 'C2')
    dot.edge('C2', 'C3')
    dot.edge('C3', 'C4')
    dot.edge('A', 'E')
    dot.edge('E', 'D')

    # Display the architecture
    dot.view()


plot_detailed_architecture()
