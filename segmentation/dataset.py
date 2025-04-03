import wfdb
import numpy as np
import os
import torch
from scipy import signal
from tqdm import tqdm
import ts2vg
from torch_geometric.data import Data
import matplotlib.pyplot as plt

## complexo QRS como sendo a região entre o fim da onda P e o início da onda T.
# Configurações
pasta_dados = "/scratch/guilherme.evangelista/ECG-Segmentation/1.0.1/data"
original_fs = 500  # Frequência de amostragem original
target_length = 4096  # Queremos exatamente 4096 pontos

# Utilizando apenas a derivação ii
lead = 'ii'

# Dicionário para mapear rótulos
label_map = {
    'p': 0,  # Onda P
    'N': 1,  # Complexo QRS (definido como a região entre o fim da P e o início da T)
    't': 2,  # Onda T
    'unlabeled': 3  # Pontos sem rótulo
}



class ECGGraph(Data):
    def __init__(self, x, edge_index, y, record_id, lead):
        super(ECGGraph, self).__init__(x=x, edge_index=edge_index, y=y)
        self.record_id = record_id
        self.lead = lead
    
    def __repr__(self):
        return f"ECGGraph(record={self.record_id}, lead={self.lead}, nodes={self.num_nodes}, edges={self.num_edges})"

def process_record(record_number, lead):
    """
    Processa um registro ECG, converte para grafo de visibilidade natural e atribui características e rótulos aos nós.
    """
    caminho_registro = os.path.join(pasta_dados, record_number)
    
    try:
        # Carregar o sinal ECG
        record = wfdb.rdrecord(caminho_registro)
        
        # Encontrar o índice da derivação desejada
        if lead in record.sig_name:
            lead_index = record.sig_name.index(lead)
            sinal_ecg = record.p_signal[:, lead_index]
        else:
            print(f"Derivação {lead} não encontrada no registro {record_number}.")
            return None
        
        # Debug: imprimir informações sobre o sinal
        print(f"\nProcessando registro {record_number}")
        print(f"Comprimento original do sinal: {len(sinal_ecg)} amostras")
        print(f"Duração original: {len(sinal_ecg)/original_fs:.2f} segundos")
        
        # Reamostrar diretamente para 4096 pontos
        sinal_ecg_resampled = signal.resample(sinal_ecg, target_length)
        
        # Debug: imprimir informações sobre o sinal reamostrado
        print(f"Comprimento após reamostragem: {len(sinal_ecg_resampled)} amostras")
        print(f"Duração após reamostragem: {len(sinal_ecg_resampled)/(target_length/10):.2f} segundos")
        
        # Calcular a primeira derivada do sinal
        # Usando diferenças finitas: dy/dx ≈ [f(x+h) - f(x)]/h
        primeira_derivada = np.zeros_like(sinal_ecg_resampled)
        primeira_derivada[:-1] = np.diff(sinal_ecg_resampled)
        primeira_derivada[-1] = primeira_derivada[-2]  # Repetir o último valor para manter o tamanho
        
        # Criar sinal invertido para o grafo de visibilidade invertido
        sinal_ecg_invertido = -sinal_ecg_resampled
        
        # Carregar anotações
        try:
            annotations = wfdb.rdann(
                caminho_registro,
                extension=lead,
                sampfrom=0,
                sampto=len(sinal_ecg)
            )
            
            # Ajustar pontos de anotação para a nova taxa de amostragem
            adjusted_samples = np.array(annotations.sample) * (target_length / len(sinal_ecg))
            annotations.sample = np.round(adjusted_samples).astype(int)
            
            # Inicializar rótulos
            labels = np.full(target_length, label_map['unlabeled'], dtype=int)
            
            # Processar anotações para encontrar intervalos
            intervalos = {'p': [], 't': []}
            inicio_atual = None
            tipo_atual = None
            
            for i, (samp, lbl) in enumerate(zip(annotations.sample, annotations.symbol)):
                if 0 <= samp < target_length:
                    if lbl == '(':
                        inicio_atual = samp
                    elif lbl in ['p', 't'] and inicio_atual is not None:  # Só processamos P e T
                        tipo_atual = lbl
                    elif lbl == ')' and inicio_atual is not None and tipo_atual is not None:
                        fim_atual = samp
                        if tipo_atual in intervalos:
                            intervalos[tipo_atual].append((inicio_atual, fim_atual))
                        inicio_atual = None
                        tipo_atual = None
            
            # Atribuir rótulos para ondas P e T
            for tipo, lista_intervalos in intervalos.items():
                for inicio, fim in lista_intervalos:
                    for i in range(inicio, fim+1):
                        if 0 <= i < len(labels):
                            labels[i] = label_map[tipo]
            
            # MODIFICAÇÃO PRINCIPAL: Identificar complexos QRS como regiões entre o fim da P e início da T
            # Organizamos os intervalos em ordem cronológica
            p_intervals = sorted(intervalos['p'])
            t_intervals = sorted(intervalos['t'])
            
            # Para cada onda P, encontramos a próxima onda T e rotulamos o intervalo entre elas como QRS
            for p_start, p_end in p_intervals:
                # Encontrar a próxima onda T após essa onda P
                next_t = None
                for t_start, t_end in t_intervals:
                    if t_start > p_end:  # A onda T começa após o fim da onda P
                        next_t = (t_start, t_end)
                        break
                
                if next_t:
                    # Rotular a região entre o fim da P e o início da T como QRS
                    for i in range(p_end + 1, next_t[0]):
                        if 0 <= i < len(labels):
                            labels[i] = label_map['N']  # Complexo QRS
            
        except Exception as e:
            print(f"Erro ao carregar anotações para o registro {record_number}, derivação {lead}: {e}")
        
        # Gerar grafo de visibilidade natural para o sinal normal
        vg_normal = ts2vg.NaturalVG().build(sinal_ecg_resampled)
        edges_normal = vg_normal.edges
        edge_index_normal = torch.tensor(np.array([[e[0], e[1]] for e in edges_normal]).T, dtype=torch.long)
        
        # Gerar grafo de visibilidade natural para o sinal invertido
        vg_invertido = ts2vg.NaturalVG().build(sinal_ecg_invertido)
        edges_invertido = vg_invertido.edges
        edge_index_invertido = torch.tensor(np.array([[e[0], e[1]] for e in edges_invertido]).T, dtype=torch.long)
        
        # Calcular graus dos nós para o grafo normal
        degrees_normal = np.zeros(target_length, dtype=int)
        for e in edges_normal:
            degrees_normal[e[0]] += 1
            degrees_normal[e[1]] += 1
        
        # Calcular graus dos nós para o grafo invertido
        degrees_invertido = np.zeros(target_length, dtype=int)
        for e in edges_invertido:
            degrees_invertido[e[0]] += 1
            degrees_invertido[e[1]] += 1
        
        # Converter para tensores
        degrees_normal = torch.tensor(degrees_normal, dtype=torch.float).view(-1, 1)
        degrees_invertido = torch.tensor(degrees_invertido, dtype=torch.float).view(-1, 1)
        
        # Características base
        amplitudes = torch.tensor(sinal_ecg_resampled, dtype=torch.float).view(-1, 1)
        derivada = torch.tensor(primeira_derivada, dtype=torch.float).view(-1, 1)
        
        # NOVA FEATURE: Coordenada X (posição do nó na sequência)
        # Criar um vetor com as posições dos nós (0 a target_length-1)
        x_coordinates = torch.tensor(np.arange(target_length), dtype=torch.float).view(-1, 1)
        
        # Normalizar coordenadas X para o intervalo [0, 1]
        x_coordinates = x_coordinates / (target_length - 1)
        
        # One-hot encoding para grafo normal (1,0)
        flag_normal = torch.ones(target_length, 1)
        flag_invertido = torch.zeros(target_length, 1)
        
        # One-hot encoding para grafo invertido (0,1)
        flag_normal_inv = torch.zeros(target_length, 1)
        flag_invertido_inv = torch.ones(target_length, 1)
        
        # Características para o grafo normal: amplitude, derivada, coordenada x, grau, flags (1,0)
        normal_features = torch.cat([
            amplitudes,
            derivada,
            x_coordinates,     # Nova feature: coordenada x
            degrees_normal,
            flag_normal,
            flag_invertido
        ], dim=1)
        
        # Características para o grafo invertido: apenas grau do grafo invertido e flags (0,1)
        invertido_features = torch.cat([
            degrees_invertido,
            flag_normal_inv,
            flag_invertido_inv
        ], dim=1)
        
        # Concatenar todas as características (agora total de 9 features por nó, em vez de 8)
        node_features = torch.cat([normal_features, invertido_features], dim=1)
        
        # Modificação: utilizar apenas o grafo normal para definir a topologia do grafo,
        # ou seja, apenas os índices de arestas do grafo normal serão mantidos.
        edge_index = edge_index_normal
        
        node_labels = torch.tensor(labels, dtype=torch.long)
        
        # Criar o objeto de grafo mantendo todas as features, mas com topologia somente do grafo normal
        graph = ECGGraph(
            x=node_features,
            edge_index=edge_index,
            y=node_labels,
            record_id=record_number,
            lead=lead
        )
        
        print(f"Grafo criado com sucesso: {target_length} nós, {edge_index.shape[1]} arestas")
        print(f"Número de características por nó: {node_features.shape[1]}")
        return graph
        
    except Exception as e:
        print(f"Erro ao processar o registro {record_number}, derivação {lead}: {e}")
        return None

def visualize_graph(graph, output_filename):
    """
    Visualiza um grafo de ECG mostrando o sinal e as conexões de visibilidade.
    """
    plt.figure(figsize=(15, 12))
    
    # Extrair dados dos nós (temos 9 características por nó agora)
    # Características do grafo normal: índices 0-5
    ecg_signal = graph.x[:, 0].numpy()          # Amplitude original
    ecg_derivada = graph.x[:, 1].numpy()        # Primeira derivada
    ecg_x_coord = graph.x[:, 2].numpy()         # Nova feature: coordenada x
    ecg_degrees_normal = graph.x[:, 3].numpy()  # Grau do nó no grafo normal (agora no índice 3)
    
    # Características do grafo invertido: índices 6-8
    # Note que agora o grau invertido está no índice 6 (em vez de 5)
    ecg_degrees_inverted = graph.x[:, 6].numpy()   # Grau do nó no grafo invertido
    
    node_labels = graph.y.numpy()
    
    # Plotar os sinais com cores baseadas nos rótulos
    plt.subplot(511)  # Mudado de 411 para 511 para acomodar o novo gráfico
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i in range(len(ecg_signal)):
        plt.plot(i, ecg_signal[i], 'o', color=colors[node_labels[i]], markersize=2)
    
    plt.title(f'ECG Signal - Record {graph.record_id}, Lead {graph.lead}')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    # Criar legenda
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[0], markersize=8, label='P wave'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[1], markersize=8, label='QRS complex'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[2], markersize=8, label='T wave'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[3], markersize=8, label='Unlabeled')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Plotar a primeira derivada
    plt.subplot(512)
    plt.plot(ecg_derivada, 'b-', linewidth=1)
    plt.title('First Derivative of ECG Signal')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    # Plotar as coordenadas X
    plt.subplot(513)
    plt.plot(ecg_x_coord, 'c-', linewidth=1)
    plt.title('X Coordinates (Normalized)')
    plt.ylabel('Coordinate Value')
    plt.grid(True)
    
    # Plotar os graus dos nós
    plt.subplot(514)
    plt.plot(ecg_degrees_normal, 'g-', label='Normal Graph', linewidth=1)
    plt.plot(ecg_degrees_inverted, 'r-', label='Inverted Graph', linewidth=1)
    plt.title('Node Degrees Comparison')
    plt.xlabel('Node Index')
    plt.ylabel('Degree')
    plt.legend()
    plt.grid(True)
    
    # Visualizar algumas arestas do grafo (limitado a 500 arestas para não sobrecarregar)
    plt.subplot(515)
    edges = graph.edge_index.numpy()
    
    # Limitar a 500 arestas para visualização
    max_edges = min(500, edges.shape[1])
    
    plt.scatter(np.arange(len(ecg_signal)), ecg_signal, c='black', s=5, alpha=0.5)
    
    for i in range(max_edges):
        src, dst = edges[0, i], edges[1, i]
        plt.plot([src, dst], [ecg_signal[src], ecg_signal[dst]], 'g-', alpha=0.1)
    
    plt.title(f'Visibility Graph Edges (showing {max_edges} out of {edges.shape[1]} edges)')
    plt.xlabel('Node Index')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Visualização salva como '{output_filename}'")

# CÓDIGO PRINCIPAL
registros = sorted(set([f.split('.')[0] for f in os.listdir(pasta_dados) if '.' in f and not f.startswith('.')]))
all_graphs = []

for record in tqdm(registros, desc="Processando registros"):
    graph = process_record(record, lead)
    if graph is not None:
        all_graphs.append(graph)

print(f"\nTotal de grafos gerados: {len(all_graphs)}")

# Estatísticas dos rótulos
label_counts = {
    'P': 0,
    'QRS': 0,
    'T': 0,
    'Unlabeled': 0
}

for graph in all_graphs:
    labels_count = torch.bincount(graph.y, minlength=4)
    label_counts['P'] += labels_count[0].item()
    label_counts['QRS'] += labels_count[1].item()
    label_counts['T'] += labels_count[2].item()
    label_counts['Unlabeled'] += labels_count[3].item()

print("\nEstatísticas de rótulos:")
print(f"  Onda P: {label_counts['P']}")
print(f"  Complexo QRS: {label_counts['QRS']}")
print(f"  Onda T: {label_counts['T']}")
print(f"  Sem rótulo: {label_counts['Unlabeled']}")

# Imprimir informações sobre as características dos nós
if len(all_graphs) > 0:
    print(f"Total: {all_graphs[0].x.shape[1]} características por nó")

# Salvar dataset
torch.save(all_graphs, 'ecg_visibility_graph_dataset.pt')
print("\nDataset salvo como 'ecg_visibility_graph_dataset.pt'")

# Visualizar o primeiro grafo como exemplo
if len(all_graphs) > 0:
    visualize_graph(all_graphs[0], 'exemplo_grafo_visibilidade.png')