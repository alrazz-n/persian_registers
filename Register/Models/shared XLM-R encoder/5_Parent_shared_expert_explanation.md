# Hierarchical XLM-R with Parent-Specific Experts

 ## 1\. Problem formulation

 The task is a **multilabel text classification problem** where labels are not independent. The labels have a natural hierarchy:

 - Some labels represent broad categories (parents).
- Other labels represent more specific subcategories (children).

 For example:

```
NA
 ├── ne
 ├── sr
 └── nb

IN
 ├── en
 ├── ra
 ├── dtp
 ├── fi
 └── lt

OP
 ├── rv
 ├── ob
 ├── rs
 └── av
```

 A document can belong to multiple categories at the same time, so this is not a single-class classification problem. The model must predict several labels simultaneously.

---

 # 2\. Flat multilabel architecture (baseline approach)

 A traditional approach would treat every label as independent.

 The architecture would look like:

```
                 XLM-R
                   |
                   |
          shared document representation
                   |
                   |
          single classification head
                   |
                   |
       --------------------------------
       |  |  |  |  |  |  |  |  | ... |
       L1 L2 L3 L4 L5 L6 L7 L8 ... L25
```

 The encoder produces one representation of the text.

 Then a single linear layer predicts all labels:

 $$
h = XLM-R(text)
$$

 $$
logits = W h + b
$$

 where:

 - `h` is the document representation.
- The classifier directly predicts all 25 labels.
- Each output neuron corresponds to one label.

 The loss is usually:

 $$
L = BCE(y, \hat{y})
$$

 where every label contributes equally.

 ## Limitation of the flat approach

 The model does not explicitly know that some labels are related.

 For example:

```
IN
 |
 +-- en
 +-- ra
 +-- dtp
 +-- fi
 +-- lt
```

 The flat classifier sees:

```
en
ra
dtp
fi
lt
```

 as five unrelated outputs.

 The model can learn correlations implicitly, but the architecture does not provide any structural information about:

 - which labels belong together,
- which labels share semantic meaning,
- which parent category a child belongs to.

 All labels compete for the same representation and the same classification layer.

---

 # 3\. Proposed architecture: hierarchical classifier

 Instead of one classifier for all labels, the model explicitly separates the prediction problem into levels.

 The architecture has three main components:

```
                 XLM-R encoder
                      |
                      |
          shared text representation (h)
                      |
        +-------------+-------------+
        |                           |
        |                           |
 Parent representation       Parent-specific experts
        |                           |
        |                           |
 Parent classifier          Child classifiers
        |
        |
 Parent predictions
```

 The encoder is still shared:

 $$
h = XLM-R(text)
$$

 The difference is what happens after this representation is created.

---

 # 4\. Parent classification branch

 First, the model predicts the high-level categories.

 The shared representation is projected into a parent-specific representation:

 $$
h_p = GELU(W_p h)
$$

 Then a parent classifier predicts the parent labels:

```
                 h
                 |
        Parent projection
                 |
          Parent classifier
                 |
       MT LY SP ID NA HI IN OP IP
```

 This produces 9 parent predictions.

 The parent task provides the model with a high-level understanding of the document.

 For example:

 The model first learns:

 > "This text belongs to the IN category."

 before trying to distinguish:

```
IN
 ├── en
 ├── ra
 ├── dtp
 ├── fi
 └── lt
```

---

 # 5\. Parent-specific expert modules

 The main difference from a flat model is the use of **experts specialized for each parent category**.

 Instead of one shared child classifier:

```
             h
             |
       one classifier
             |
       all children
```

 the model has separate expert networks:

```
                    h
                    |
       --------------------------------
       |              |               |
       |              |               |
    NA expert      IN expert       OP expert
       |              |               |
       |              |               |
  ne sr nb      en ra dtp fi lt   rv ob rs av
```

 Each expert receives the same XLM-R representation but learns transformations specialized for its parent category.

 For example:

 ### NA expert

 Only learns:

```
ne
sr
nb
```

 ### IN expert

 Only learns:

```
en
ra
dtp
fi
lt
```

 ### OP expert

 Only learns:

```
rv
ob
rs
av
```

 The expert acts as a specialized feature extractor for its own group of labels.

---

 # 6\. Why use experts instead of one child classifier?

 In a flat architecture:

```
                 h
                 |
        ----------------
        |              |
       IN labels     OP labels
        |              |
       en             rv
       ra             ob
       fi             rs
```

 The same classifier parameters are responsible for all categories.

 However, the linguistic patterns needed to identify different categories may be different.

 For example:

 - Some child labels may depend on vocabulary.
- Some may depend on writing style.
- Some may depend on discourse patterns.
- Some may depend on register-specific signals.

 The expert modules allow each group of related labels to learn its own transformation:

 $$
z_{IN}=Expert_{IN}(h)
$$

 $$
z_{OP}=Expert_{OP}(h)
$$

 $$
z_{NA}=Expert_{NA}(h)
$$

 Then each expert has a small classifier:

 $$
child_{IN}=W_{IN}z_{IN}+b
$$

---

 # 7\. Important difference: no hard routing

 A possible hierarchical model would route a document only through the predicted parent:

 Example:

```
Parent prediction:

IN = 1
NA = 0
OP = 0

Only use IN expert
```

 However, this model does **not** do that.

 Because this is multilabel classification, a document may belong to multiple parent categories.

 Therefore:

 - Every expert always receives the shared representation.
- There is no decision gate.
- All child classifiers are active.

 The model learns all parent-child relationships jointly.

---

 # 8\. Loss function

 The training objective contains two learning signals:

 ## Parent loss

 The model learns broad categories:

 $$
L_{parent}=BCE(parent\ logits,parent\ labels)
$$

 ## Child loss

 The model learns fine-grained categories:

 $$
L_{child}=BCE(child\ logits,child\ labels)
$$

 The final loss is:

 $$
L =
\lambda_p L_{parent}
+
\lambda_c L_{child}
$$

 where:

 - $\lambda_p$ controls how important parent prediction is.
- $\lambda_c$ controls how important child prediction is.

 These weights are optimized using Optuna.

---

 # 9\. How the two architectures differ

 | Aspect | Flat multilabel classifier | Hierarchical expert model |
| --- | --- | --- |
| Encoder | Shared XLM-R | Shared XLM-R |
| Representation | One shared representation | One shared representation |
| Classification | One head for all labels | Separate parent and child branches |
| Label relationship | Learned implicitly | Explicit hierarchy |
| Child labels | All treated equally | Grouped by parent |
| Parameters | One classifier | Parent classifier + specialized experts |
| Knowledge sharing | Fully shared | Shared encoder + specialized modules |
| Parent information | Not explicitly used | Parent prediction guides representation learning |
| Output | 25 independent logits | 9 parent logits \+ 16 child logits |

---

 # 10\. Intuition in one sentence

 A flat model asks:

 > "Given this text, which of these 25 labels are active?"

 The hierarchical expert model asks:

 > "First understand the broad categories of this text, then use specialized experts to determine the specific labels inside each category."

---

 # 11\. Short presentation version

 If you need to explain it quickly:

 > I built a hierarchical multilabel classifier on top of XLM-R. Instead of using one classification head for all labels, the model separates prediction into parent categories and child categories. The shared XLM-R encoder produces a document representation. One branch predicts the parent labels, while separate expert networks specialize in predicting the children belonging to each parent category. The model is trained with a combined parent and child loss, allowing it to learn both the global structure of the label space and the fine-grained distinctions between related labels. Unlike a flat classifier, the architecture explicitly incorporates label hierarchy instead of treating all labels as independent outputs.

 This explanation should be suitable for a paper presentation, thesis defense, or discussion with someone familiar with neural classification architectures.