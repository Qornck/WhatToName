### 2025.05.09
更改了邻接矩阵的写法
{'precision': [np.float64(0.05476320087098529), np.float64(0.03897659227000543), np.float64(0.025830157866086), np.float64(0.016521502449646164)], 'recall': [np.float64(0.22999236292649472), np.float64(0.3261416228482097), np.float64(0.4205493109684726), np.float64(0.5368906542559239)], 'ndcg': [np.float64(0.16328498187977722), np.float64(0.19367300491502326), np.float64(0.21789848764197936), np.float64(0.2389342802444491)]}

todo: 
- [x] 加diffusion model
- [x] 同构图节点选择

### 2025.05.12
{'precision': [np.float64(0.054763200870985296), np.float64(0.038649972781709306), np.float64(0.026020685900925423), np.float64(0.01686173108328797)], 'recall': [np.float64(0.23100292920652202), np.float64(0.3264263680431345), np.float64(0.42603015746728323), np.float64(0.545583657859107)], 'ndcg': [np.float64(0.16412152375217592), np.float64(0.18981570360779934), np.float64(0.21508676491621415), np.float64(0.2425569463408236)]}
hyperparameters:
![alt text](image.png)

### 2025.05.13
diffusion graph的topk选择不同 作为对比学习的两个视图

{'precision': [np.float64(0.053892215568862284), np.float64(0.03810560696788242), np.float64(0.02555797495917256), np.float64(0.01657593903102885)], 'recall': [np.float64(0.23558118528178407), np.float64(0.3238228837031232), np.float64(0.41751123128368645), np.float64(0.5367778927659167)], 'ndcg': [np.float64(0.15627081643695642), np.float64(0.18862448725041894), np.float64(0.2159956481436253), np.float64(0.23515398101821364)]}

6 6 0.1 0.13

{'precision': [np.float64(0.0522591181273816), np.float64(0.03777898747958628), np.float64(0.025421883505715843), np.float64(0.016657593903102882)], 'recall': [np.float64(0.22798169893978276), np.float64(0.32327502836484867), np.float64(0.42018769653500193), np.float64(0.5405975262262688)], 'ndcg': [np.float64(0.15541587458720832), np.float64(0.18234701531173012), np.float64(0.21482648470717097), np.float64(0.24030553079136527)]}

6 6 0.1 0.1

### 2025.05.15
添加cold-start和warm-start数据的研究
- [x] warm-start
- [ ] cold-start

### 2025.05.19
- [x] GNN中，尝试同构图和异构图分别用不同层数的gnn,只在最后一层进行融合

### 2025.05.27
adjacent matrix 删去部分 
{'precision': [np.float64(0.05280348394120849), np.float64(0.03837778987479586), np.float64(0.02610234077299946), np.float64(0.01664398475775721)], 'recall': [np.float64(0.2221297664411437), np.float64(0.32485159551027815), np.float64(0.4248182954769781), np.float64(0.5403344160829191)], 'ndcg': [np.float64(0.15405144781383687), np.float64(0.18732404553468415), np.float64(0.21247356256133737), np.float64(0.2358263666125872)]}

### 2025.06.12
折腾半天似乎又回到最开始了

### 2025.06.25
tmux 0-2:
```
getSparseGraph:
# adj_mat[:self.n_mashups, :self.n_mashups] = mmR
# adj_mat[self.n_mashups:, self.n_mashups:] = aaR

getDiffSparseGraph:
adj_mat[:self.n_mashups, :self.n_mashups] = mmR
adj_mat[self.n_mashups:, self.n_mashups:] = aaR
```

tmux 3:
```
getSparseGraph:
adj_mat[:self.n_mashups, :self.n_mashups] = mmR
adj_mat[self.n_mashups:, self.n_mashups:] = aaR

getDiffSparseGraph:
adj_mat[:self.n_mashups, :self.n_mashups] = mmR
adj_mat[self.n_mashups:, self.n_mashups:] = aaR
```

### 2025.06.26
