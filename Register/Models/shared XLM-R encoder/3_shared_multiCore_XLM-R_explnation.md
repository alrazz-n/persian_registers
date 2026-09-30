# Hierarchical Multilabel Classification with XLM-R

## Overview

The model uses XLM-R as a shared encoder...
The encoder parameters are updated by both classification objectives. Therefore, the learned document representation is influenced by both coarse-grained category recognition and fine-grained subtype recognition.
The dataset contains:

- 9 parent labels
- 16 child labels
- 25 total output labels


## Architecture

The architecture consists of two classification heads:

\[
\text{Total Loss} =
\lambda_p L_{parent}
+
\lambda_c L_{child}
\]

where:

- \(L_{parent}\) is the parent classification loss
- \(L_{child}\) is the child classification loss
- \(\lambda_p\) and \(\lambda_c\) are loss weights

 ## 1\. Problem formulation

 The task is a **multilabel text classification problem** where each document can belong to multiple categories at the same time.

 The labels are not completely independent. They have a **hierarchical structure**:

 - There are **high-level categories (parents)**.
- Some of these parents have more specific **subcategories (children)**.

 For example:

```
IN (Internet)
 ├── en
 ├── ra
 ├── dtp
 ├── fi
 └── lt
```

 A document classified as `en` should also belong to the broader category `IN`.

 Similarly:

```
NA
 ├── ne
 ├── sr
 └── nb
```

 The hierarchy contains semantic information: child labels describe more specific cases inside parent categories.

 The goal is therefore not only to predict labels, but to learn the relationship between general and specific categories.

---

 ## 2. Baseline approach: flat multilabel classification

 A standard way to solve this problem would be a **flat multilabel classifier**.

 The architecture would look like this:

```
                 XLM-R Encoder
                       |
              document representation
                       |
              Single classification head
                       |
              25 independent outputs
```

 The model produces one logit for every label:

```
[MT, LY, SP, ID, NA, HI, IN, OP, IP,
 it, ne, sr, nb, re, en, ra, dtp, fi, lt,
 rv, ob, rs, av, ds, ed]
```

 Each output is treated independently:

```
Probability(MT)
Probability(LY)
Probability(SP)
...
Probability(en)
Probability(ra)
```

 The model learns:

 > "Given this text representation, what is the probability of each label?"

 The loss is calculated over all labels together using multilabel binary classification loss.

 The advantage of this approach is simplicity:

 - One encoder
- One classification head
- One prediction vector

 However, the model does not explicitly know that:

```
en → IN
ne → NA
dtp → IN
```

 The relationship between labels must be learned indirectly from data.

 For example, the model may learn that `en` often appears together with `IN`, but there is no architectural constraint saying that these labels are connected.

---

 ## 3\. Proposed approach: hierarchical classification

 Instead of treating all labels as unrelated, I split the prediction task into two related problems.

 The architecture is:

```
                       XLM-R Encoder
                             |
                    Shared representation
                             |
              +--------------+--------------+
              |                             |
              v                             v

      Parent classifier             Child classifier

        9 outputs                    16 outputs

 MT LY SP ID NA HI IN OP IP       it ne sr nb re ...
```

 The encoder is shared between both tasks.

 The XLM-R encoder reads the text and creates a contextual representation of the document.

 This representation is then used by two classification heads:

 ### Parent classification head

 Predicts broad categories:

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

 ### Child classification head

 Predicts fine-grained categories:

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

 The final output is still 25 labels, but internally the model learns two different levels of abstraction.

---

 ## 4\. Why use two classification heads?

 The main motivation is that parent and child labels represent different levels of information.

 Parent labels answer:

 > "What general type of register or category does this document belong to?"

 Child labels answer:

 > "What specific subtype does this document belong to?"

 These are related but different prediction tasks.

 The parent classifier encourages the encoder to learn broad semantic signals.

 The child classifier focuses on fine-grained distinctions.

 The shared encoder receives learning signals from both:

```
                 XLM-R Encoder

              /                 \
             /                   \

     Parent loss             Child loss

   General information     Detailed information
```

 The total training objective combines both:

```
Total loss =
(parent loss × parent weight)
+
(child loss × child weight)
```

 Therefore, the model learns representations that are useful for both hierarchical levels.

### Training objective

The model is trained using a combined loss:

$$
L_{total}
=
\lambda_p L_{parent}
+
\lambda_c L_{child}
$$

where:

| Symbol | Meaning |
|---|---|
| \(L_{parent}\) | Loss for predicting parent labels |
| \(L_{child}\) | Loss for predicting child labels |
| \(\lambda_p\), \(\lambda_c\) | Weights controlling the contribution of each loss |

---

 ## 5\. How the hierarchy is used during training

 The hierarchy is also reflected in the labels.

 If a child label is present, the parent label is activated.

 For example:

 Original annotation:

```
en = 1
```

 is transformed into:

```
IN = 1
en = 1
```

 because `en` belongs to `IN`.

 The model therefore learns the natural dependency:

```
child presence implies parent presence
```

This encourages the model to produce hierarchy-consistent predictions by training it with labels where child categories imply their corresponding parent categories.

```
en = true
IN = false
```

 which would violate the hierarchy.

---

 ## 6\. Main difference between flat and hierarchical approaches

 | Aspect | Flat multilabel classifier | Hierarchical classifier |
| --- | --- | --- |
| Classification heads | One | Two |
| Label relationship | Implicit | Explicit |
| Parent-child dependency | Learned only from data | Built into architecture |
| Prediction levels | All labels treated equally | General and specific levels |
| Encoder | Shared | Shared |
| Output labels | 25 | 25 |
| Loss | One multilabel loss | Parent loss + child loss |

---

 ## 7\. Example

 Imagine a document about a specific internet communication phenomenon.

 A flat classifier sees:

```
Document representation
        |
        |
25 independent decisions

IN = 0.91
en = 0.87
ra = 0.65
NA = 0.15
...
```

 The model learns correlations between labels but has no explicit structure.

 The hierarchical model sees:

```
Document representation
        |
        |
        +----------------+
        |                |
        v                v

 Parent prediction    Child prediction

 IN = 0.91            en = 0.87
                      ra = 0.65
```

 The parent head focuses on recognizing the general domain, while the child head specializes in identifying the subtype.

---

 ## 8\. Why this architecture can be beneficial

 The main expected advantages are:

 ### 1\. Better use of label structure

 The model knows that some labels are related instead of independent.

 ### 2\. Better learning for rare child labels

 Fine-grained labels often have fewer examples.

 The parent task provides an additional learning signal about the broader category.

 ### 3\. More interpretable predictions

 Instead of only saying:

```
en = true
```

 the model also explains:

```
IN = true
 └── en = true
```

 which follows the hierarchy.

 ### 4\. Multi-task learning effect

 The two heads act as related tasks.

 Learning to predict parents can improve the shared representation used for children.




---

 ## 9\. Summary explanation (short version for presentations)

 > I implemented a hierarchical multilabel classifier based on XLM-R. Instead of treating all 25 labels as independent outputs, I separated them into two levels: parent categories and child categories. The encoder generates a shared text representation, which is then used by two classification heads: one predicts broad categories and the other predicts fine-grained subcategories. The training objective combines both parent and child classification losses, allowing the model to learn both general and specific patterns. Compared with a flat multilabel classifier with one head, this architecture explicitly incorporates the label hierarchy and encourages predictions that are consistent with the parent-child relationships defined by the label hierarchy.

---

 This explanation should be suitable for a thesis, paper discussion, or explaining the model design to colleagues.

 