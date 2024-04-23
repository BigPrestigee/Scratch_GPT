import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
batch_size = 16 # batch size
block_size = 32 # what is the maximum context length for predictions????
max_epochs = 20000 # number of epochs
eval_interval = 100 # how often to evaluate the model
learning_rate = 1e-4 # learning rate
device = 'cuda' if torch.cuda.is_available() else 'cpu' # use GPU if available
eval_iters = 200 # number of iterations to evaluate the model
n_embd = 64 
n_head = 4
n_layer = 4
dropout = 0.0

# define the model
torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

"""
在你提供的例子中，@torch.no_grad()是PyTorch库中的一个装饰器。这个装饰器的作用是在其下方的函数(仅对下方的第一个函数起作用)
执行时，禁用梯度计算。在需要进行模型推理而不需要进行反向传播（比如，在模型评估时）时使用，可以减少内存消耗并加速计算。
"""
@torch.no_grad()
def evaluate_loss():
    out = {}  # 创建一个空字典，用于存储不同数据分割（'train', 'val'）的平均损失。
    model.eval()  # 将模型设置为评估模式，通常意味着禁用dropout和batch normalization等。

    # 循环遍历 'train' 和 'val' 数据集。
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)  # 初始化一个张量，用于存储每次迭代的损失。
        
        # 对于每个数据集，执行eval_iters次迭代来计算损失。
        for k in range(eval_iters):
            X, Y = get_batch(split)  # 获取一批数据。
            logits, loss = model(X, Y)  # 通过模型传递数据以获取预测结果（logits）和损失值。
            losses[k] = loss.item()  # 把当前迭代的损失值存储到losses张量中。

        # 计算当前数据集分割的平均损失，并将其添加到out字典中。
        out[split] = losses.mean()

    model.train()  # 模型回到训练模式，使得dropout和batch normalization等再次生效。
    return out  # 返回包含'train'和'val'平均损失的字典。

# single head attention
class Head(nn.Module):
    """
        single head
    """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False) # y = ax + b, if bias = False, then b = 0
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        """
        self.tril == tensor([[1, 0, 0, 0],
                             [1, 1, 0, 0],
                             [1, 1, 1, 0],
                             [1, 1, 1, 1]])
        """

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # B, T ,C -> B, T, H
        q = self.query(x) # B, T, C -> B, T, H
        # compute the attention scores 
        wei = q @ k.transpose(-2, -1) / (C ** 0.5) # (B,T,H) * (B,H,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # weighted aggregation of values
        v = self.value(x)
        out = wei @ v # (B, T, T) * (B, T, H) -> (B, T, H)
        return out
    
# multi-head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        """
            N 个头, (B, T, H) -> (B, T, N * H)
            N * H should == n_embd
        """
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # paper -> Wo 拼接之后在进行一次映射,实现跨头的信息交互和权重参数化
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
            ???dim = -1
        """
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# feed forward network    
class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """
        transformer block
    """
    def __init__(self, n_embd, n_head):
        super().__init__()
        # split the n_embd into n_head
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.foward = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        """
            这里采用pre-norm的结构：LayerNorm -> Attention -> Residual -> LayerNorm -> FeedForward -> Residual
            优点：防止底层信息被淹没，即较大的方差可能会使得淹没底层信息（较小的差异）
        """
        x = x + self.sa(self.ln1(x))
        x = x + self.foward(self.ln2(x))
        return x
    
# super simple bigram model
class BigramLLM(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # attention block
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        # 为啥从n_embd到vocab_size？？64->65
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets = None):
        # batch size, context length
        """
            inputs:
            torch.Size([4, 8])
            tensor([[24, 43, 58,  5, 57,  1, 46, 43],
                    [44, 53, 56,  1, 58, 46, 39, 58],
                    [52, 58,  1, 58, 46, 39, 58,  1],
                    [25, 17, 27, 10,  0, 21,  1, 54]])
            targets:
            torch.Size([4, 8])
            tensor([[43, 58,  5, 57,  1, 46, 43, 39],
                    [53, 56,  1, 58, 46, 39, 58,  1],
                    [58,  1, 58, 46, 39, 58,  1, 46],
                    [17, 27, 10,  0, 21,  1, 54, 39]])
        """
        B, T = idx.shape
        tok_embeddings = self.token_embedding_table(idx) # B, T -> B, T, C
        # [0,...,T-1] -> embeddings -> T, C
        pos_embeddings = self.position_embedding_table(torch.arange(T, device=device)) # T -> T, C
        # broadcast the position embeddings to all batches
        x = tok_embeddings + pos_embeddings
        # x-> self attention
        x = self.blocks(x)
        # final layer norm
        """
            为啥最终要进行一次层归一化？
        """
        x = self.ln_f(x)
        logits = self.lm_head(x) # B, T, C -> B, T, vocab_size

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape # B, T, vocab_size？
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # 选择每一行最后block_size个元素
            idx_cond = idx[:, -block_size:]
            # prediction (idx = idx_cond, target = None)
            logits, loss = self(idx_cond) # B, T, vocab_size
            # 取每一行最后一个元素的embedding, 即取取得是43, 58, 1, 54 的embedding
            """
                B, T = 4, 8
                tensor([[24, 43, 58,  5, 57,  1, 46, 43],
                        [44, 53, 56,  1, 58, 46, 39, 58],
                        [52, 58,  1, 58, 46, 39, 58,  1],
                        [25, 17, 27, 10,  0, 21,  1, 54]])
                B, T, vocab_size = 4, 8, 65
            """
            logits = logits[:, -1, :] # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)
            # sample a token
            idx_next = torch.multinomial(probs, num_samples=1) # # (B, 1)
            idx = torch.cat((idx, idx_next), dim=-1) # (B, T+1)

        return idx
    
model = BigramLLM()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M')

# optimizer
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

for iter in range(max_epochs):
    
    # print loss every eval_interval iterations
    if iter % eval_interval == 0 or iter == max_epochs - 1:
        losses = evaluate_loss()
        print(f'step {iter}, train loss: {losses["train"]:.4f}, val loss: {losses["val"]:.4f}')
    
    # train 
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optim.zero_grad()
    loss.backward()
    optim.step()

# generate some text
context = torch.zeros((1, 1), dtype=torch.long).to(device)
# 返回batch_size中的第一行元素
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))