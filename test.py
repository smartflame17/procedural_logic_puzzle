# full_generator_with_layers.py
import random
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
from enum import Enum

# ============================================================
# AST node definitions
# ============================================================
class Node:
    pass

@dataclass(frozen=True)
class Pred(Node):
    name: str   # 'T', 'I', 'L'
    arg: str

@dataclass(frozen=True)
class Not(Node):
    x: Node

@dataclass(frozen=True)
class And(Node):
    left: Node
    right: Node

@dataclass(frozen=True)
class Or(Node):
    left: Node
    right: Node

@dataclass(frozen=True)
class Xor(Node):
    left: Node
    right: Node

@dataclass(frozen=True)
class Imply(Node):
    left: Node
    right: Node

@dataclass(frozen=True)
class Iff(Node):
    left: Node
    right: Node

# ============================================================
# Tokenizer + Parser
# ============================================================
TOKEN_SPEC = [
    ("ARROW",    r"->"),
    ("IFF",      r"<->"),
    ("NOT",      r"!"),
    ("AND",      r"&"),
    ("XOR",      r"\^"),
    ("OR",       r"\|"),
    ("LPAREN",   r"\("),
    ("RPAREN",   r"\)"),
    ("COMMA",    r","),
    ("IDENT",    r"[A-Za-z_][A-Za-z0-9_]*"),
    ("WS",       r"[ \t\n\r]+"),
]

TOKEN_RE = re.compile("|".join(f"(?P<{k}>{v})" for k, v in TOKEN_SPEC))

@dataclass
class Token:
    kind: str
    text: str

def tokenize(s: str) -> List[Token]:
    tokens: List[Token] = []
    for m in TOKEN_RE.finditer(s):
        kind = m.lastgroup
        text = m.group()
        if kind == "WS":
            continue
        tokens.append(Token(kind, text))
    tokens.append(Token("EOF", ""))
    return tokens

class ParseError(Exception):
    pass

class Parser:
    def __init__(self, tokens: List[Token]):
        self.toks = tokens
        self.i = 0

    def peek(self) -> Token:
        return self.toks[self.i]

    def eat(self, kind: str) -> Token:
        t = self.peek()
        if t.kind != kind:
            raise ParseError(f"Expected {kind}, got {t.kind} ({t.text})")
        self.i += 1
        return t

    def try_eat(self, kind: str) -> Optional[Token]:
        if self.peek().kind == kind:
            return self.eat(kind)
        return None

    def parse_expr(self) -> Node:
        node = self.parse_imply()
        while self.try_eat("IFF"):
            right = self.parse_imply()
            node = Iff(node, right)
        return node

    def parse_imply(self) -> Node:
        node = self.parse_xor()
        while self.try_eat("ARROW"):
            right = self.parse_xor()
            node = Imply(node, right)
        return node

    def parse_xor(self) -> Node:
        node = self.parse_or()
        while self.try_eat("XOR"):
            right = self.parse_or()
            node = Xor(node, right)
        return node

    def parse_or(self) -> Node:
        node = self.parse_and()
        while self.try_eat("OR"):
            right = self.parse_and()
            node = Or(node, right)
        return node

    def parse_and(self) -> Node:
        node = self.parse_unary()
        while self.try_eat("AND"):
            right = self.parse_unary()
            node = And(node, right)
        return node

    def parse_unary(self) -> Node:
        if self.try_eat("NOT"):
            return Not(self.parse_unary())
        return self.parse_primary()

    def parse_primary(self) -> Node:
        tok = self.peek()
        if tok.kind == "IDENT":
            pred_name = self.eat("IDENT").text
            if self.try_eat("LPAREN"):
                arg = self.eat("IDENT").text
                self.eat("RPAREN")
                if pred_name not in ("T", "I", "L"):
                    raise ParseError(f"Unknown predicate '{pred_name}'. Use T, I, or L.")
                return Pred(pred_name, arg)
            else:
                raise ParseError(f"Unexpected IDENT '{pred_name}'. Expected predicate like T(A)")
        elif tok.kind == "LPAREN":
            self.eat("LPAREN")
            node = self.parse_expr()
            self.eat("RPAREN")
            return node
        else:
            raise ParseError(f"Unexpected token: {tok.kind} '{tok.text}'")

def parse_formula(s: str) -> Node:
    return Parser(tokenize(s)).parse_expr()

# ============================================================
# AST -> text and evaluator
# ============================================================
def ast_to_str(n: Node) -> str:
    if isinstance(n, Pred):
        return f"{n.name}({n.arg})"
    if isinstance(n, Not):
        x = n.x
        inner = ast_to_str(x)
        if isinstance(x, Pred) or isinstance(x, Not):
            return f"!{inner}"
        return f"!({inner})"
    if isinstance(n, And):
        return f"{ast_to_str(n.left)} & {ast_to_str(n.right)}"
    if isinstance(n, Or):
        return f"{ast_to_str(n.left)} | {ast_to_str(n.right)}"
    if isinstance(n, Xor):
        return f"{ast_to_str(n.left)} ^ {ast_to_str(n.right)}"
    if isinstance(n, Imply):
        return f"{ast_to_str(n.left)} -> {ast_to_str(n.right)}"
    if isinstance(n, Iff):
        return f"{ast_to_str(n.left)} <-> {ast_to_str(n.right)}"
    raise TypeError(f"Unknown node type: {type(n)}")

# Given the node and a world (traitor, liar_map), evaluate truth
def eval_ast(n: Node, traitor: str, liar_map: Dict[str, bool]) -> bool:
    if isinstance(n, Pred):
        if n.name == "T":
            return n.arg == traitor
        if n.name == "I":
            return n.arg != traitor
        if n.name == "L":
            return bool(liar_map.get(n.arg, False))
        raise ValueError("Unknown predicate")
    if isinstance(n, Not):
        return not eval_ast(n.x, traitor, liar_map)
    if isinstance(n, And):
        return eval_ast(n.left, traitor, liar_map) and eval_ast(n.right, traitor, liar_map)
    if isinstance(n, Or):
        return eval_ast(n.left, traitor, liar_map) or eval_ast(n.right, traitor, liar_map)
    if isinstance(n, Xor):
        return eval_ast(n.left, traitor, liar_map) ^ eval_ast(n.right, traitor, liar_map)
    if isinstance(n, Imply):
        a = eval_ast(n.left, traitor, liar_map)
        b = eval_ast(n.right, traitor, liar_map)
        return (not a) or b
    if isinstance(n, Iff):
        return eval_ast(n.left, traitor, liar_map) == eval_ast(n.right, traitor, liar_map)
    raise TypeError(f"Unknown node type: {type(n)}")

# ============================================================
# Evidence and Clue node dataclasses
# ============================================================
@dataclass
class EvidenceNode:
    _id_counter: int = field(init=False, default=0, repr=False)
    text: str = field(default="")
    id: int = field(init=False)

    def __post_init__(self):
        type(self)._id_counter += 1
        object.__setattr__(self, "id", type(self)._id_counter)

@dataclass
class ClueNode:
    _id_counter: int = field(init=False, default=0, repr=False)
    ast: Node = field(default=None)
    text: str = field(default="")
    is_goal: bool = field(default=False)
    connections: List["ClueNode"] = field(default_factory=list)  # points toward closer-to-goal nodes (parents)
    evidences: List[EvidenceNode] = field(default_factory=list)
    layer: int = field(default=0)  # distance-from-goal
    subject: List[str] = field(default=None)   
    reference: List[str] = field(default=None) 
    id: int = field(init=False)

    def __post_init__(self):
        type(self)._id_counter += 1
        object.__setattr__(self, "id", type(self)._id_counter)

    def __repr__(self):
        goal_tag = " GOAL" if self.is_goal else ""
        return f"Clue#{self.id}({self.text}{goal_tag}, subj={self.subject}, ref={self.reference})"
    
# Utility to extract subject and reference from an AST
# For simplicity: if Imply -> left side = reference(s), right side = subject(s).
# Direct (Pred/Not) -> subject = reference = NPC involved.
#TODO: handle extraction into 2 lists of npcs (subject, reference)
def extract_subject_reference(ast):
    # helper to collect all npc args from a subtree
    def collect_refs(node):
        if isinstance(node, Pred):
            return [node.arg]
        if isinstance(node, Not):
            return collect_refs(node.x)
        if isinstance(node, And):
            return collect_refs(node.left) + collect_refs(node.right)
        return [], []
    def collect_subj(node):
        if isinstance(node, Pred):
                return [node.arg]
        if isinstance(node, Not):
            return collect_subj(node.x)
        if isinstance(node, (And, Or, Xor)):
             return collect_subj(node.left) + collect_subj(node.right)
        return [], []
    

    if isinstance(ast, (Pred, Not)):
        # direct clue about single npc
        if isinstance(ast, Pred):
            return [ast.arg], [ast.arg]
        else: # Not node
            if isinstance(ast.x, Pred):
                return [ast.x.arg], [ast.x.arg]
            return [], []
    if isinstance(ast, Imply):
        # reference(s) = npc(s) from left
        ref_npcs = collect_refs(ast.left)
        # subject(s) = npc(s) from right
        subj_npcs = collect_subj(ast.right)
        return subj_npcs, ref_npcs
    # Fallback: if Or/Xor top-level, just treat all NPCs as subjects and references //TODO
    if isinstance(ast, (Or, Xor)):
        return collect_subj(ast), collect_subj(ast)
    return [], []

# ============================================================
# Formula factories (structured AST producers)
# ============================================================
def make_direct(npcs: List[str]) -> Node:
    # direct facts: T(A) or !T(A) or I(A) or !I(A)
    npc = random.choice(npcs)
    if random.random() < 0.6:
        atom = Pred("T", npc)
    else:
        atom = Pred("I", npc)
    if random.random() < 0.4:
        return Not(atom)
    return atom

def make_conditional(npcs: List[str], use_liars: bool) -> Node:
    # form: (literal) -> (literal or conjunction)
    a = random.choice(npcs)
    b = random.choice([x for x in npcs if x != a])
    # antecedent often uses I(a) or T(a)
    ant = Pred("I", a) if random.random() < 0.7 else Pred("T", a)
    if random.random() < 0.5:
        cons = Not(Pred("T", b))
    else:
        cons = Pred("T", b)
    # sometimes make antecedent a conjunction of two literals
    if random.random() < 0.25 and len(npcs) >= 3:
        c = random.choice([x for x in npcs if x not in (a, b)])
        ant = And(ant, Pred("I", c))
    return Imply(ant, cons)

def make_multi(npcs: List[str], use_liars: bool) -> Node:
    # disjunctions / xor / small combos over 2-3 npcs
    k = min(3, len(npcs))
    sample = random.sample(npcs, k)
    left = Pred("T", sample[0])
    right = Pred("T", sample[1]) if len(sample) > 1 else Pred("T", sample[0])
    # most multi are ambiguous: use XOR or OR
    if random.random() < 0.6:
        node = Xor(left, right)
    else:
        node = Or(left, right)
    # occasionally add a small implication wrapper to increase ambiguity
    if random.random() < 0.2 and len(sample) >= 3:
        extra = Not(Pred("T", sample[2]))
        node = Imply(extra, node)
    return node

def make_complex(npcs: List[str], use_liars: bool) -> Node:
    # somewhat more complex compound rules used rarely
    if random.random() < 0.5:
        return make_conditional(npcs, use_liars)
    return make_multi(npcs, use_liars)

# ============================================================
# Priority system: ClueType and chooser
# ============================================================
class ClueType(Enum):
    DIRECT = 3
    CONDITIONAL = 2
    MULTI = 1
    COMPLEX = 1  # treated similar to MULTI for priority

def choose_clue_type(distance_from_goal: int, max_depth: int) -> ClueType:
    """
    Return a ClueType biased by distance_from_goal:
      - far from goal (large distance) -> prefer MULTI/CONDITIONAL
      - close to goal (small distance) -> prefer DIRECT
    We calculate a weight per type: weight = base_priority * depth_bias
    depth_bias = 1 / (distance_from_goal + 1)  (closer to 0 increases weight for high-priority)
    We'll invert so that closer layers (distance small) amplify DIRECT weight.
    """
    # base priorities
    base = {
        ClueType.DIRECT: 3.0,
        ClueType.CONDITIONAL: 2.0,
        ClueType.MULTI: 1.0,
        ClueType.COMPLEX: 1.0,
    }
    # depth factor: we want DIRECT to become relatively stronger when distance is small
    # use factor = (max_depth - distance + 1) / (max_depth + 1)
    factor = (max_depth - distance_from_goal + 1) / (max_depth + 1)
    weights = {}
    for t, p in base.items():
        # bias: direct scaled directly by factor, multi scaled inversely
        if t == ClueType.DIRECT:
            weights[t] = p * (0.6 + 0.8 * factor)  # boost toward top layers
        else:
            weights[t] = p * (1.6 - 0.8 * factor)  # decrease toward top layers
    total = sum(weights.values())
    choices, w = zip(*[(k, v) for k, v in weights.items()])
    probs = [v / total for v in w]
    return random.choices(choices, weights=probs, k=1)[0]

# ============================================================
# Consistency/triviality checks
# ============================================================
def is_true_under_world(n: Node, traitor: str, liar_map: Dict[str, bool]) -> bool:
    return eval_ast(n, traitor, liar_map)

def is_tautology_over_traitor_choices(n: Node, npcs: List[str], liar_map: Dict[str, bool]) -> bool:
    for t in npcs:
        if not eval_ast(n, t, liar_map):
            return False
    return True

def is_contradiction_over_traitor_choices(n: Node, npcs: List[str], liar_map: Dict[str, bool]) -> bool:
    for t in npcs:
        if eval_ast(n, t, liar_map):
            return False
    return True

# ============================================================
# Utility to create a random usable formula AST according to chosen type
# ============================================================
def make_formula_of_type(ctype: ClueType, npcs: List[str], use_liars: bool) -> Node:
    if ctype == ClueType.DIRECT:
        return make_direct(npcs)
    elif ctype == ClueType.CONDITIONAL:
        return make_conditional(npcs, use_liars)
    elif ctype == ClueType.MULTI:
        return make_multi(npcs, use_liars)
    else:
        return make_complex(npcs, use_liars)

# ============================================================
# Evidence helper
# ============================================================
def new_evidence(npcs: List[str]) -> EvidenceNode:
    who = random.choice(npcs)
    flavor = random.choice(["camera log", "alibi", "testimony", "forensics", "keycard", "phone GPS"])
    return EvidenceNode(text=f"{who} {flavor}")

# ============================================================
# Generator: layer-based graph builder
# ============================================================
def generate_puzzle_with_layers(
    npcs: List[str],
    num_clues: int = 12,
    max_depth: int = 4,
    num_chains: int = 3,
    liar_count: int = 0,
    seed: Optional[int] = None,
) -> Dict[str, Union[str, ClueNode, List[ClueNode], List[EvidenceNode], Dict[str, bool]]]:

    if seed is not None:
        random.seed(seed)

    # Basic parameters
    traitor = random.choice(npcs)
    liars = random.sample(npcs, k=min(liar_count, len(npcs)))
    liar_map = {x: (x in liars) for x in npcs}
    use_liars = liar_count > 0

    # Create goal node at layer 0
    goal_ast = Pred("T", traitor)
    goal_node = ClueNode(ast=goal_ast, text=ast_to_str(goal_ast), is_goal=True, layer=0, subject=[traitor], reference=[traitor])

    all_clues = [goal_node]
    seen_texts = {goal_node.text}
    evidences: List[EvidenceNode] = []
    used_direct_subjects = set()    # track subjects used in direct clues to avoid duplicates

    # Distribute target number of clue nodes across layers 1..max_depth
    # We'll create roughly num_clues nodes excluding goal
    target_nodes = max(1, num_clues)
    # allocate by a simple distribution: more nodes in middle layers
    layer_counts = [0] * (max_depth + 1)
    # ensure at least one node in the farthest layer if possible (gives early-game content)
    for d in range(1, max_depth + 1):
        # weight for d (farther layers get slightly more)
        w = 1 + (d / (max_depth + 1)) * 1.5
        layer_counts[d] = w
    # normalize to integer counts
    total_w = sum(layer_counts[1:])
    for d in range(1, max_depth + 1):
        layer_counts[d] = max(1, int(round((layer_counts[d] / total_w) * target_nodes)))

    # adjust to exact target (messy but it works lol)
    cur_sum = sum(layer_counts[1:])
    while cur_sum > target_nodes:
        # remove from largest non-empty layer
        idx = max(range(1, max_depth + 1), key=lambda i: layer_counts[i])
        if layer_counts[idx] > 1:
            layer_counts[idx] -= 1
            cur_sum -= 1
        else:
            break
    while cur_sum < target_nodes:
        idx = max(range(1, max_depth + 1), key=lambda i: -layer_counts[i])
        layer_counts[idx] += 1
        cur_sum += 1

    # We'll generate layer by layer, outermost first (farthest from goal), because player sees far ones first
    # Connections must point toward closer layers (smaller layer numbers)
    layers: Dict[int, List[ClueNode]] = {d: [] for d in range(max_depth + 1)}   # start with empty list per layer (0..max_depth)
    layers[0] = [goal_node] # goal at layer 0

    # Pre-seed chains: make sure there are num_chains independent roots in the farthest layers
    # We'll generate nodes per layer and connect each new node to a random node in the previous (closer) layer.
    for d in range(max_depth, 0, -1):  # start from farthest layer up to 1
        want = layer_counts[d]  # how many nodes to create at this layer
        created = 0
        attempts = 0
        while created < want and attempts < want * 30:
            attempts += 1
            # choose type biased by distance
            ctype = choose_clue_type(d, max_depth)

            # attempt candidate formulas until valid
            for _try in range(120):
                break_loop = False
                ast_candidate = make_formula_of_type(ctype, npcs, use_liars)
                text = ast_to_str(ast_candidate)

                # uniqueness
                if text in seen_texts:
                    continue
                # must be true under the real world
                if not is_true_under_world(ast_candidate, traitor, liar_map):
                    continue
                # should not be trivially true/false across all traitor choices
                if is_tautology_over_traitor_choices(ast_candidate, npcs, liar_map):
                    continue
                if is_contradiction_over_traitor_choices(ast_candidate, npcs, liar_map):
                    continue

                # good candidate
                # extract subject and reference
                subj, ref = extract_subject_reference(ast_candidate)
                # if chosen type is DIRECT, ensure no duplicate subjects
                if ctype == ClueType.DIRECT:                    
                    for npc in subj:  # should be only one npc in direct clues
                        if npc == traitor:
                            break_loop = True
                            break  # skip direct clue that spoils traitor
                        if npc in used_direct_subjects:
                            break_loop = True
                            break  # skip duplicate direct clue
                    
                    if break_loop:
                        continue    # restart outer loop

                    used_direct_subjects.add(npc)
                    node = ClueNode(ast=ast_candidate, text=text, layer=d, subject=subj, reference=ref)
                else:
                    node = ClueNode(ast=ast_candidate, text=text, layer=d, subject=subj, reference=ref)

                print(f"[Debug] Created clue at layer {d}: {text} (type={ctype.name}, subj={subj}, ref={ref})")

                # connect to random node in closer layers (0..d-1), prefer closer (small index) //TODO: connect according to subject/reference relevance
                possible_parents = []
                for pd in range(0, d):
                    possible_parents.extend(layers[pd])
                if not possible_parents:
                    # fallback: connect to goal
                    node.connections.append(goal_node)
                else:
                    parent = random.choice(possible_parents)
                    node.connections.append(parent)
                layers[d].append(node)
                all_clues.append(node)
                seen_texts.add(text)
                created += 1
                break

            # finished candidate attempts
        # ensure at least some nodes exist at the layer; if nothing created, add a simple true literal related to traitor (fallback)
        if created == 0:
            fb = Pred("I", random.choice([x for x in npcs if x != traitor]))
            fb_text = ast_to_str(fb)
            if fb_text not in seen_texts:
                node = ClueNode(ast=fb, text=fb_text, layer=d)
                node.connections.append(random.choice(layers[0]))
                layers[d].append(node)
                all_clues.append(node)
                seen_texts.add(fb_text)
            print(f"[Debug] Fallback clue at layer {d}: {fb_text}")

    # Add optional merging: randomly rewire some nodes to point to non-immediate parents to make multiple chains merge.
    for node in [n for d in range(1, max_depth + 1) for n in layers[d]]:
        if random.random() < 0.22:
            # pick another parent from any closer layer (0..node.layer-1)
            cand_parents = []
            for pd in range(0, node.layer):
                cand_parents.extend(layers[pd])
            if cand_parents:
                maybe = random.choice(cand_parents)
                if maybe not in node.connections:
                    node.connections.append(maybe)

    # Attach evidences to all clues
    for clue in all_clues:
        need = random.randint(1, 3)
        for _ in range(need):
            ev = new_evidence(npcs)
            clue.evidences.append(ev)
            evidences.append(ev)

    # Add some noise (extra true but not in chain) nodes modestly
    noise_nodes = []
    noise_target = max(0, num_clues // 4)
    noise_attempts = 0
    while len(noise_nodes) < noise_target and noise_attempts < noise_target * 40:
        noise_attempts += 1
        d = random.randint(1, max_depth)  # put noise at some layer
        ctype = choose_clue_type(d, max_depth)
        for _try in range(120):
            break_loop = False
            ast_candidate = make_formula_of_type(ctype, npcs, use_liars)
            text = ast_to_str(ast_candidate)
            if text in seen_texts:
                continue
            if not is_true_under_world(ast_candidate, traitor, liar_map):
                continue
            if is_tautology_over_traitor_choices(ast_candidate, npcs, liar_map):
                continue
            if is_contradiction_over_traitor_choices(ast_candidate, npcs, liar_map):
                continue

            subj, ref = extract_subject_reference(ast_candidate)
            # if chosen type is DIRECT, ensure no duplicate subjects & no spoiler for traitor
            if ctype == ClueType.DIRECT:                    
                for npc in subj:  # should be only one npc in direct clues
                    if npc == traitor:
                        break_loop = True
                        break  # skip direct clue that spoils traitor
                    if npc in used_direct_subjects:
                        break_loop = True
                        break  # skip duplicate direct clue

                if break_loop:
                    continue    # mark subject as used and restart outer loop

                used_direct_subjects.add(npc)
                node = ClueNode(ast=ast_candidate, text=text, layer=d, subject=subj, reference=ref)
            else:
                node = ClueNode(ast=ast_candidate, text=text, layer=d, subject=subj, reference=ref)

            print(f"[Debug] Created noise clue at layer {d}: {text} (type={ctype.name}, subj={subj}, ref={ref})")
            # connect to a random closer node
            parent_pool = []
            for pd in range(0, d):
                parent_pool.extend(layers[pd])
            if parent_pool:
                node.connections.append(random.choice(parent_pool))
            else:
                node.connections.append(goal_node)
            node.evidences.append(new_evidence(npcs))
            noise_nodes.append(node)
            all_clues.append(node)
            seen_texts.add(text)
            layers[d].append(node)
            break

    # final shuffle of all_clues for presentation randomness
    random.shuffle(all_clues)

    return {    # return dict of results => will be a class in C# conversion
        "traitor": traitor, # str(name) of traitor npc
        "liars": liar_map,  # dict of npc -> bool
        "goal": goal_node,  # ClueNode of goal
        "clues": all_clues, # list of all ClueNode (used for iteration)
        "layers": layers,   # dict of layer_num -> list of ClueNode (used for searching by layer)
        "evidences": evidences, # list of all EvidenceNode
        "noise": noise_nodes,   # list of noise ClueNode (used for finding noise quickly)
    }

# ============================================================
# Example usage (tweak parameters here)
# ============================================================
if __name__ == "__main__":
    # Example parameters you can tune:
    npcs = ["A", "B", "C", "D", "E"]
    puzzle = generate_puzzle_with_layers(
        npcs=npcs,
        num_clues=10,   # target number of additional clue nodes (approx)
        max_depth=4,    # how many layers between earliest clues and goal
        num_chains=3,   # used implicitly (distribution)
        liar_count=1,   # set >0 to enable liar predicates
        seed=111
    )

    # quick debug print (concise)
    print(f"Traitor: {puzzle['traitor']}")
    print(f"Liars: {[k for k,v in puzzle['liars'].items() if v]}")
    print("\nClues (showing id, layer, text, connections, evidence count, ref and subj of the text):")
    for c in sorted(puzzle["clues"], key=lambda x: (x.layer, x.id)):
        parents = [f"Clue#{p.id}" for p in c.connections]
        print(f"Clue#{c.id} || layer={c.layer} || {c.text} || parent-> {parents} || evidences={len(c.evidences)} || ref={c.reference} || subj={c.subject}")