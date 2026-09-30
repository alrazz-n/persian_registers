  # Hierarchical Hard-Routed Expert Model

 ## Motivation

 A flat multi-label classifier treats every label as an independent prediction problem. If there are 25 labels, the model learns 25 independent output probabilities:

```
                 XLM-R
                   |
                   |
              shared representation
                   |
                   |
          +---------------------+
          |                     |
          ▼                     ▼
        Label 1              Label 2
        Label 2              Label 3
        Label 3              Label 4
        ...
        Label 25
```

 The classifier does not explicitly know that some labels belong together or that some labels represent broader categories while others represent subcategories.

 For example, if:

```
IN
 ├── en
 ├── ra
 ├── dtp
 ├── fi
 └── lt
```

 then a flat classifier sees:

```
IN
en
ra
dtp
fi
lt
```

 as six unrelated labels.

 It must independently learn that `en`, `ra`, `dtp`, `fi`, and `lt` are related because they share a common parent.

---

 # Proposed Architecture

 The proposed model introduces hierarchy into the classification process.

 Instead of one classifier predicting everything, the model has two levels:

 1. A **parent classifier**
2. Multiple **parent-specific child experts**

 The overall architecture is:

```
                    XLM-R
                      |
                      |
              shared representation h
                      |
          +-----------+-----------+
          |                       |
          ▼                       ▼
 Parent classifier          Parent-specific experts
          |
          |
  predicts broad domains
```

 The parent classifier predicts the high-level categories:

```
NA
HI
IN
OP
IP
SP
...
```

 Each active parent then activates its own expert responsible only for predicting its children.

 Example:

```
Parent prediction:

IN = active
NA = active
OP = inactive

Routing:

          IN expert
             |
             |
        en ra dtp fi lt

          NA expert
             |
             |
          ne sr nb
```

 The model does not send every example through every expert. It only activates the experts corresponding to the detected parent categories.

---

 # Training Strategy: Gold Routing

 During training, the routing decision uses the true parent labels.

 Example:

 The training example has:

```
Parent labels:

IN = 1
NA = 1
OP = 0
```

 The routing becomes:

```
                 XLM-R
                    |
                    |
              representation h
                    |
                    |
          Parent classifier
                    |
                    |
        -----------------------
        |                     |
        ▼                     ▼
    NA expert             IN expert
        |                     |
        ▼                     ▼
    ne sr nb             en ra dtp fi lt
```

 The important point is that the child experts receive training examples even if the parent classifier is not yet good.

 Without this strategy, a problem occurs:

```
Parent prediction wrong
          |
          ↓
Expert not activated
          |
          ↓
Child receives no training signal
```

 Using gold routing avoids this early training problem because the correct expert always receives examples during training.

---

 # Inference Strategy: Predicted Routing

 During validation and testing, the model behaves realistically.

 The parent classifier predictions determine routing.

 Example:

 The model predicts:

```
NA = 0.91
IN = 0.83
OP = 0.12
```

 with threshold 0.5:

```
NA → active
IN → active
OP → inactive
```

 Therefore:

```
NA expert runs
IN expert runs
OP expert is skipped
```

 Only these experts generate child predictions.

---

 # Why Use Experts?

 The main motivation is specialization.

 A flat classifier has one decision boundary for all labels:

```
                 XLM-R
                    |
                    |
              one classifier
                    |
 ------------------------------------------------
 |    |    |    |    |    |    |    |    |       |
NA   HI   IN   OP   IP   SP   ne   sr   en ...
```

 The same classifier parameters must separate very different types of labels.

 The hierarchical model instead divides the problem:

```
                 XLM-R
                    |
                    |
              parent classifier
                    |
      --------------------------------
      |              |               |
      ▼              ▼               ▼
   NA expert      IN expert       OP expert

      |              |               |
      ▼              ▼               ▼

  ne sr nb      en ra dtp       rv ob rs av
```

 Each expert only learns distinctions among related labels.

 For example:

 The IN expert only needs to distinguish:

```
en vs ra vs dtp vs fi vs lt
```

 It does not need to consider unrelated labels such as:

```
ne
sr
rv
ob
```

 This reduces competition between unrelated labels.

---

 # Difference From a Flat Architecture

 ## Flat model

 A flat model learns:

```
text → 25 independent probabilities
```

 Advantages:

 - Simple architecture
- Easy training
- No routing errors
- Every label always receives information

 Disadvantages:

 - Ignores label relationships
- Rare labels compete with unrelated labels
- The classifier must learn all distinctions simultaneously

---

 ## Hierarchical hard-routed model

 The proposed model learns:

```
text
 |
 |
parent decision
 |
 |
select relevant expert
 |
 |
predict children
```

 Advantages:

 - Uses known label structure
- Allows specialization
- Reduces the search space for child prediction
- Prevents unrelated labels from competing directly
- Gives each group of labels a dedicated representation transformation

 Disadvantages:

 - Introduces dependence on parent predictions
- Parent mistakes can block child predictions during inference
- Training and inference routing distributions are different because training uses gold parents while inference uses predicted parents

---

 # Main Difference in One Sentence

 A flat classifier asks:

 > "Given this text, which of the 25 labels are active?"

 The hierarchical expert model asks:

 > "First determine which broad categories are relevant, then use specialized classifiers to decide which specific labels inside those categories apply."

---

 # How to Describe the Training/Test Difference

 A concise explanation:

 > During training, I use gold parent labels for routing so that every child expert receives the correct examples and can learn independently of early parent classification errors. During inference, I switch to predicted parent routing to simulate the real deployment scenario, where the model must first decide which experts to activate. This creates a realistic hierarchical pipeline while avoiding the cold-start problem during training.

---

 # Short Presentation Version

 If explaining it in a meeting:

 > "Instead of using one flat classifier for all labels, I introduced a hierarchy. The model first predicts broad parent categories using a multi-label parent classifier. These predictions determine which specialized child experts are activated. Each expert only predicts the fine-grained labels belonging to its parent. During training, I use gold parents to route examples so child experts receive enough learning signal. During testing, routing is based on predicted parents, making the system behave like a real hierarchical classifier. The goal is to exploit label relationships and allow specialized classifiers to focus on smaller, related label groups rather than forcing one classifier to learn all distinctions simultaneously."