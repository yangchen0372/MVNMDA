import numpy as np
class Data_Helper():
    def __init__(self,version=2):

        if version == 2:
            path = r'miRNA-disease association file path'
            num_miRNA = 495
            num_disease = 383
            self.known_md = np.loadtxt(path, dtype=int) - 1
        elif version == 3 or version==3.2:
            path = r'miRNA-disease association file path'
            num_miRNA = 788
            num_disease=374
            self.known_md = np.loadtxt(path, dtype=int) - 1

        self.num_of_known_association = self.known_md.shape[0]
        self.num_of_unknown_association = num_miRNA * num_disease - self.num_of_known_association

        self.miRNA_feature = np.zeros((num_miRNA, num_miRNA+num_disease))
        self.disease_feature = np.zeros((num_disease, num_miRNA+num_disease))

        self.Adj = np.zeros((num_miRNA, num_disease), dtype=int)
        for i in range(self.known_md.shape[0]):
            self.Adj[self.known_md[i, 0], self.known_md[i, 1]] = 1
            self.miRNA_feature[self.known_md[i, 0], self.known_md[i, 1]] = 1
            self.disease_feature[self.known_md[i, 1], self.known_md[i, 0]] = 1

        miRNA_Adj = np.eye(num_miRNA)
        disease_Adj = np.eye(num_disease)
        self.md_Adj = np.vstack([
            np.hstack([miRNA_Adj, self.Adj]),
            np.hstack([self.Adj.T, disease_Adj])
        ])
        self.md_feature = np.vstack([
            self.miRNA_feature.copy(),
            self.disease_feature.copy()
        ])
        self.md_pos_edge_index = np.argwhere(self.md_Adj == 1)
        self.md_neg_edge_index = np.argwhere(self.md_Adj == 0)
        np.savetxt(r'out_file_path', self.md_feature, fmt='%.4f')

        self.mdm_Adj = self.MetaPathSearch(self.Adj)
        self.mdm_feature = self.miRNA_feature.copy()
        self.mdm_pos_edge_index = np.argwhere(self.mdm_Adj == 1)
        self.mdm_neg_edge_index = np.argwhere(self.mdm_Adj == 0)

        self.dmd_Adj = self.MetaPathSearch(self.Adj.T)
        self.dmd_feature = self.disease_feature.copy()
        self.dmd_pos_edge_index = np.argwhere(self.dmd_Adj == 1)
        self.dmd_neg_edge_index = np.argwhere(self.dmd_Adj == 0)

        self.md_ones_Adj = np.ones((self.md_Adj.shape))
        self.md_ones_feature = np.vstack([
            self.miRNA_feature.copy(),
            self.disease_feature.copy()
        ])
        self.md_ones_pos_edge_index = np.argwhere(self.md_ones_Adj == 1)
        self.md_ones_neg_edge_index = np.argwhere(self.md_ones_Adj == 0)

    def MetaPathSearch(Adj):
        rows, cols = Adj.shape
        meta_path_graph = np.zeros((rows,rows))
        for row in range(rows):
            head_index = row
            for col in range(cols):
                mid_index = col
                if Adj[row, col] == 1:
                    tails_index = np.nonzero(Adj[:, col])[0]
                    for tail_index in tails_index.tolist():
                        meta_path_graph[head_index,tail_index] = 1
        return meta_path_graph