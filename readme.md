## 프로토타입 논리 퍼즐 생성기

### AST
기초적인 논리 연산자 지원 (AND, OR, NOT, XOR, ->, <->)  
토큰은 Node 클래스 기반 확장, 퍼즐의 가장 기본 단위가 됨

### Tokeniser + Parser
기초적인 Lexical Analysis 제공  
```ast_to_str``` 메소드 통해 직관적인 string으로 변환  
```eval_ast``` 메소드 통해 T/F 판별  

### Generator
Parameters
```
npcs: NPC 이름 모음 리스트
num_clues: 생성할 총 노드 개수
max_depth: AST의 최대 깊이
num_chains: 최대 분기 루트 (현재 미사용)
liar_count: 거짓말쟁이 NPC
seed: 재생성 위한 시드
```

**1. 사전 준비** 

시드 기반 '배신자' 및 거짓말쟁이 NPC 결정, 배신자 NPC 노드 layer 0에 저장  

**2. 가중치 기반 노드 분배**  

```max_depth``` 기반 각 layer 가중치 ```1 + (d / (max_depth + 1)) * 1.5``` 부여 후 normalize  
비는 layer 없도록 노드 분배

**3. 노드 생성 / 논리 체크 + 부모 노드 지정**  

시드 기반 랜덤 ClueNode 생성 후, 다음 메소드 통해 진리표 체크   
```is_true_under_world``` 메소드가 recursive하게 진리표 (배신자, 거짓말쟁이 NPC) 확인   
```is_tautology_over_traitor_choices``` 메소드가 해당 Node가 항상 참인지 확인  ex) T(A) ^ !T(A)   
```is_contradiction_over_traitor_choices``` 메소드가 해당 Node가 모순인지 확인  ex) T(A) & !T(A)  

해당 검사를 모두 통과한 노드는 좌변과 우변에서 담당하는 NPC 추출 후, 그래프에 포함  
대상 NPC 하나에 대한 논리 노드의 경우 중복 검사 시행  
부모 노드의 경우 상위 layer에서 임의로 결정 >> 추후 담당 NPC 기반으로 변경 예정

**4. 노이즈 추가**  

정답에 직접적인 도움이 안 되는 ClueNode를 3과 같은 방식으로 추가 

<br>  

### TODO
___
증거 노드 생성 추가  
부모 노드 연결 로직 개선  
fallback 로직 개선  
초기 노드 내용 개선 (부모 노드 연결 로직 개선 시 해결될 수도?)