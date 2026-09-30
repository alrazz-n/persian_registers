 # Hierarchical XLM-R with Parent-Based Hard Routing and Specialized Experts

 ## 1\. Problem formulation

 The task is a **hierarchical multi-label classification problem**.

 The labels are not independent. They have a structure:

 - There are **high-level parent categories** (for example: `NA`, `IN`, `OP`, `HI`).
- Some parents contain more specific **child categories** (for example:
  - `NA → {ne, sr, nb}`
  - `IN → {en, ra, dtp, fi, lt}`
  - `OP → {rv, ob, rs, av}`).

 A document can belong to multiple parent categories at the same time. Therefore, this is not a single-label classification problem. It is a **multi-label hierarchical prediction problem**.

 The model has to answer two related questions:

 1. **Which broad categories are relevant?**
   - Example:
     - Is this document related to `NA`?
     - Is it related to `IN`?
2. **Which specific subcategories inside those active categories apply?**
   - If `IN` is active:
     - Is it `en`?
     - Is it `ra`?
     - Is it `fi`?

---

 # 2\. Overall architecture

 The model is built on top of XLM-R, which acts as a shared multilingual encoder.

 The input text first goes through XLM-R:

```
Text
 |
 |
XLM-R encoder
 |
 |
Document representation
```

 The resulting representation is used by two different components:

 1. A **parent classifier**
2. A set of **parent-specific child experts**

 The architecture can be viewed as:

```
                 Input text
                     |
                    XLM-R
                     |
          ------------------------
          |                      |
   Parent classifier        Parent routing
          |                      |
 Parent probabilities     Active parents
          |                      |
          ------------------------
                     |
          Parent-specific experts
                     |
              Child classifiers
                     |
             Child predictions
```

---

 # 3\. Parent prediction stage

 The first stage predicts the parent categories.

 The model produces independent probabilities for every parent:

 Example:

```
NA = 0.91
IN = 0.87
OP = 0.12
HI = 0.05
```

 Because this is multi-label classification, several parents can be active simultaneously.

 A threshold is applied:

```
probability >= threshold → active
```

 Example with threshold 0.5:

```
NA = active
IN = active
OP = inactive
HI = inactive
```

 The active parents determine which child experts are used.

---

 # 4\. Hard routing mechanism

 The main idea of this architecture is that **not every child classifier is used for every example**.

 Instead of sending every document through all child classifiers, the model activates only the experts associated with predicted parents.

 Example:

 The parent classifier predicts:

```
NA = active
IN = active
OP = inactive
```

 The routing becomes:

```
             Document representation

                     |
        -----------------------------
        |                           |
     NA expert                  IN expert
        |                           |
   NA children                IN children
        |                           |
 ne, sr, nb                en, ra, dtp, fi, lt
```

 The OP expert is never used because OP was not activated.

 This creates a **conditional computation path**.

 The model dynamically decides which parts of the network should process each example.

---

 # 5\. Parent-specific experts

 Each parent category has its own expert network.

 For example:

```
NA expert:
    learns representations useful for:
        ne
        sr
        nb

IN expert:
    learns representations useful for:
        en
        ra
        dtp
        fi
        lt
```

 The motivation is that different parent categories may contain different linguistic patterns.

 A flat classifier assumes that all labels can be predicted from the same representation in the same way.

 This architecture allows each group of related labels to have a specialized prediction module.

---

 # 6\. Training process

 During training, the model follows the same mechanism as inference.

 The input goes through:

```
Text
 |
XLM-R
 |
Parent classifier
 |
Parent probabilities
 |
Threshold
 |
Hard routing
 |
Experts
 |
Child predictions
```

 The child loss is calculated only for the experts that were activated.

 For example:

 If the model predicts:

```
NA active
IN active
OP inactive
```

 then:

 - NA child classifier receives training signal.
- IN child classifier receives training signal.
- OP child classifier does not process that example.

 The parent classifier is trained separately with its own loss.

 The total objective is:

```
Total loss =
      Parent classification loss
      +
      Child classification loss
```

 with adjustable weights between the two.

---

 # 7\. Difference from a flat architecture

 A flat architecture would look like this:

```
                 Input text

                     |
                   XLM-R

                     |

              Single classifier

                     |

      --------------------------------
      |  |  |  |  |  |  |  |  |       |
      25 independent label outputs
```

 The classifier directly predicts every label:

```
MT
LY
SP
ID
NA
HI
IN
OP
IP
it
ne
sr
nb
re
en
ra
...
```

 All labels share the same final prediction layer.

---

 # 8\. Main differences

 ## Flat model

 A flat model treats every label as an independent decision.

 Example:

```
Probability:
NA = 0.90
IN = 0.85
en = 0.80
fi = 0.75
OP = 0.10
rv = 0.05
```

 The model predicts all labels directly.

 The classifier must learn:

 - parent-child relationships,
- label dependencies,
- differences between broad and specific categories,

 implicitly.

---

 ## Hierarchical routed model

 The hierarchy is explicitly included.

 The model first learns:

```
Is this document related to this parent category?
```

 Then:

```
Given this parent, which child labels apply?
```

 The decision process becomes:

```
Parent decision
       |
       |
 Child decision inside active parents
```

 The model does not spend equal effort on unrelated child labels.

---

 # 9\. Advantages of the hierarchical approach

 ## 1\. Uses label structure

 The model knows that:

```
en belongs to IN
ne belongs to NA
rv belongs to OP
```

 This information is directly encoded into the architecture.

 A flat model does not explicitly know these relationships.

---

 ## 2\. Specialized learning

 Each expert focuses on a smaller semantic space.

 Instead of one classifier learning 16 different child categories together:

```
Flat:

one classifier → all children

Hierarchical:

NA expert → NA children
IN expert → IN children
OP expert → OP children
```

---

 ## 3\. Conditional computation

 For each example, only relevant experts are activated.

 Example:

 A document routed to `IN` does not need to use the `OP` expert.

 This reduces unnecessary computation and allows specialization.

---

 ## 4\. Better handling of label imbalance

 In many hierarchical datasets, some labels are rare.

 A flat classifier has to learn rare labels together with all other labels.

 The hierarchical model gives rare child labels a more focused environment inside their parent category.

---

 # 10\. Trade-offs compared with a flat classifier

 The hierarchical model introduces additional complexity.

 ## Advantages:

 - Uses known label relationships.
- Allows expert specialization.
- Reduces competition between unrelated labels.
- Provides interpretable routing decisions.

 ## Challenges:

 - Errors can propagate.

 For example:

 If the parent classifier predicts:

```
IN = inactive
```

 then the IN children cannot be predicted, even if one of them should have been active.

 This is called **routing error propagation**.

 A flat classifier does not have this problem because all labels are predicted independently.

---

 # 11\. Summary explanation (short version)

 You can explain it like this:

 > "Instead of using one classifier that predicts all labels independently, I designed a hierarchical model where the prediction process follows the label structure. XLM-R first produces a document representation. A parent classifier predicts which high-level categories are relevant. These predictions are then used as a routing mechanism to activate only the corresponding parent-specific experts. Each expert specializes in predicting the child labels belonging to that parent. Therefore, the model performs classification in two stages: first selecting relevant label groups, then making fine-grained predictions inside those groups. Compared with a flat classifier, this approach explicitly models label hierarchy and allows specialized learning for different parts of the label space, but it also introduces the possibility of routing errors when an incorrect parent decision prevents a child label from being considered."

 This is the core idea of your implementation without going through the code details.