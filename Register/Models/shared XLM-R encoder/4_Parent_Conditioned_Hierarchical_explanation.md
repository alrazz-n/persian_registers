 # Hierarchical Parent-Conditioned XLM-R Multilabel Classification

 ## 1\. The problem setup

 The task is a **multilabel classification problem** where each text can belong to multiple categories.

 The labels have a natural hierarchy:

 - Some labels represent **high-level categories (parents)**.
- Other labels represent **more specific categories (children)** that belong under those parents.

 For example:

```
Parent category:
    IN (Interpersonal)

Child categories:
    en
    ra
    dtp
    fi
    lt
```

 A child label is not independent from the parent label. If a text is classified as a child category, it should also usually imply the presence of its parent category.

 Therefore, instead of treating all labels as unrelated, the model explicitly learns this hierarchy.

---

 # 2\. Flat multilabel architecture (baseline approach)

 A standard flat multilabel classifier would look like this:

```
                 Text
                   |
                   v
                XLM-R
                   |
                   v
        Shared representation h
                   |
                   v
          Single classification head
                   |
                   v
          25 independent logits
```

 The model produces one output for every label:

```
[MT, LY, SP, ID, NA, HI, IN, OP, IP,
 it, ne, sr, nb, re, en, ra, dtp, fi, lt, rv, ob, rs, av, ds, ed]
```

 The classifier learns:

 - Is MT present?
- Is LY present?
- Is SP present?
- Is it present?
- Is en present?
- etc.

 Every label is treated as an independent binary decision.

 The loss is usually:

 $$
L = BCE(label_1,...,label_{25})
$$

 The model does not explicitly know that:

```
en → IN
fi → IN
ra → IN
```

 It can learn these relationships implicitly from the data, but the architecture does not enforce or represent them.

---

 # 3\. Why introduce hierarchy?

 The main idea is that the parent categories contain useful semantic information for predicting the children.

 A child classifier should not only ask:

 > "What information does the text representation contain?"

 It should also ask:

 > "What parent-level category does this text appear to belong to?"

 For example, distinguishing between two child categories may require knowing the broader context first.

 Instead of predicting all labels at the same level, the model first builds a parent-level understanding and then uses that information for child prediction.

---

 # 4\. The proposed hierarchical architecture

 The architecture has three main components:

```
                    Text
                     |
                     v
                   XLM-R
                     |
                     v
             Shared representation h
                     |
          +----------+----------+
          |                     |
          v                     |
 Parent projection              |
          |                     |
          v                     |
 Parent representation          |
          |                     |
          v                     |
 Parent classifier              |
          |                     |
          v                     |
 Parent predictions             |
                                |
          +---------------------+
                     |
                     v
          Child classifier
                     |
                     v
             Child predictions
```

 The XLM-R encoder creates a shared representation:

 $$
h
$$

 For XLM-R-large:

 $$
h \in R^{1024}
$$

 This representation contains the general information extracted from the text.

---

 # 5\. Parent representation learning

 The model creates a special parent representation:

 $$
h_{parent}=GELU(W h+b)
$$

 The purpose of this layer is to transform the general XLM-R representation into a representation specialized for parent classification.

 Instead of using the original representation directly, the model learns:

 > "What information from the text is important for deciding the broad category?"

 The resulting vector is:

 $$
h_{parent}\in R^{256}
$$

 This is a compressed representation of parent-level information.

---

 # 6\. Parent classifier

 The parent classifier predicts the high-level categories:

```
parent_hidden
      |
      v
Parent classifier
      |
      v
9 parent logits
```

 The model predicts:

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
```

 These predictions are learned using a multilabel binary classification loss.

---

 # 7\. Conditioning the child classifier

 The main difference from a flat model is that the child classifier receives additional hierarchical information.

 The child classifier does not only receive:

 $$
h
$$

 Instead it receives:

 $$
[h ; h_{parent}]
$$

 where:

 - $h$ = original XLM-R representation
- $h_{parent}$ = learned parent representation

 Therefore the input becomes:

```
Original text representation
+
Parent-level semantic representation
```

 In your model:

```
1024 dimensions
+
256 dimensions

=
1280 dimensional child representation
```

 Then the child classifier predicts:

```
it
ne
sr
nb
re
en
ra
dtp
fi
lt
rv
ob
rs
av
ds
ed
```

---

 # 8\. Important design choice: soft hierarchy, not hard hierarchy

 A possible alternative would be:

 1. Predict parents.
2. Take the predicted parent labels.
3. Use them as input to the child classifier.

 For example:

```
Parent classifier
        |
        v
[0,0,1,0,0,1,0,0,0]
        |
        v
Child classifier
```

 However, this creates problems:

 - Parent mistakes are passed directly to the child classifier.
- The decision becomes non-differentiable if hard predictions are used.
- The child classifier depends on imperfect parent decisions.

 Your architecture avoids this.

 Instead of using hard parent predictions, it uses:

 $$
h_{parent}
$$

 The child classifier receives a continuous learned representation.

 This means:

 - Information flows from parent learning to child learning.
- The entire model remains end-to-end trainable.
- The child classifier can use useful parent information without being forced by parent errors.

---

 # 9\. Joint hierarchical loss

 The model learns both tasks simultaneously.

 The total loss is:

 $$
L =
\lambda_p L_{parent}
+
\lambda_c L_{child}
$$

 where:

 - $L_{parent}$ = error in parent predictions
- $L_{child}$ = error in child predictions
- $\lambda_p$ = importance of parent task
- $\lambda_c$ = importance of child task

 These two weights are optimized using Optuna.

 The idea is to find the balance between:

 - learning the broad categories,
- and learning the detailed categories.

 For example:

 If parent loss is too important:

```
The model may become good at broad categories
but ignore fine distinctions.
```

 If child loss dominates:

```
The model may learn specific labels
without using the hierarchy effectively.
```

---

 # 10\. Main difference compared with flat classification

 | Aspect | Flat multilabel model | Hierarchical model |
| --- | --- | --- |
| Output structure | All labels treated equally | Parents and children separated |
| Classifier heads | One classifier head | Parent head + child head |
| Label relationship | Learned implicitly | Explicitly represented |
| Child prediction | Based only on XLM-R features | Uses XLM-R + parent representation |
| Training | One multilabel loss | Weighted parent + child loss |
| Hierarchical information | Not directly available | Injected into child prediction |

---

 # 11\. Intuition

 A flat model thinks:

 > "I have 25 labels. I need to decide each one independently."

 Your hierarchical model thinks:

 > "First understand the general type of the text. Then use that understanding to make more specific decisions."

 The model first learns:

```
What kind of text is this?
```

 and then:

```
Given that context, which specific subcategories apply?
```

---

 # 12\. One-sentence explanation

 A concise explanation for presentations:

 > "I built a hierarchical multilabel XLM-R classifier where the encoder produces a shared text representation, one branch learns parent-level categories, and the learned parent representation is combined with the original representation to condition the child classifier. Unlike a flat classifier that predicts all labels independently, this architecture explicitly models the relationship between broad categories and their subcategories while keeping the entire model trainable end-to-end."