import torch
import re
import io

def g_disable_dropout(node):
    if isinstance(node, torch.nn.ModuleList) or isinstance(node, list):
        for sub_node in node:
            g_disable_dropout(sub_node)
    elif hasattr(node, "__typename") and re.match("Dropout", node.__typename):
        node.train = False

def g_enable_dropout(node):
    if isinstance(node, torch.nn.ModuleList) or isinstance(node, list):
        for sub_node in node:
            g_enable_dropout(sub_node)
    elif hasattr(node, "__typename") and re.match("Dropout", node.__typename):
        node.train = True

def g_clone_many_times(net, T):
    clones = []
    mem = io.BytesIO()
    torch.save(net, mem)
    
    for _ in range(T):
        mem.seek(0)  # Reset buffer pointer
        clone = torch.load(mem)
        
        # Ensure parameters & gradients match original net
        with torch.no_grad():
            torch.nn.utils.vector_to_parameters(
                torch.nn.utils.parameters_to_vector(net.parameters()), clone.parameters()
            )
            torch.nn.utils.vector_to_parameters(
                torch.nn.utils.parameters_to_vector(net.parameters()), clone.parameters()
            )
        
        clones.append(clone)

    return clones

def g_init_gpu(args):
    gpuidx = args[0] if args else 1
    print(f"Using {gpuidx}-th GPU")
    torch.cuda.set_device(gpuidx - 1)  # Convert to 0-based index
    g_make_deterministic(1)

def g_make_deterministic(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Ensures multi-GPU consistency
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.synchronize()  # Ensures operations complete before moving on

def g_replace_table(to, from_):
    assert len(to) == len(from_), "Mismatched tensor lists"
    for i in range(len(to)):
        to[i].copy_(from_[i])

def g_f3(f):
    return "{:.3f}".format(f)

def g_d(f):
    return "{}".format(int(round(f)))
