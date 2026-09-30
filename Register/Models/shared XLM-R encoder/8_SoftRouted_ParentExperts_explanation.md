 # Hierarchical XLM-R with Parent-Specific Expert Routing

 ## Motivation

 The task is a **multi-label classification problem** where labels have a natural hierarchy.

 Instead of treating all labels as independent categories, the labels are organized into two levels:

 1. **Parent labels**\
    These represent broad categories.
2. **Child labels**\
    These represent more specific subcategories inside each parent.

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

 A document may belong to multiple parents at the same time, and each active parent may have multiple child labels.

 The idea is that the model should first understand the **general type of the text**, and then use that information to make more specialized predictions.

---

 # Flat Architecture Baseline

 A traditional flat multi-label classifier would look like this:

```
                XLM-R Encoder
                     |
                     |
              Document representation
                     |
                     |
             Single classification head
                     |
                     |
        --------------------------------
        | | | | | | | | | | | | | | |
       MT LY SP ID NA HI IN OP IP ne ...
```

 The model receives the text, creates one representation, and predicts all labels simultaneously.

 For example, if there are 25 labels:

```
XLM-R representation
          |
          v
Linear layer
          |
          v
25 independent logits
          |
          v
25 sigmoid probabilities
```

 Each label has its own output neuron.

 The model learns:

 > "Given this text representation, what is the probability of each label?"

 The problem is that the classifier does not explicitly know that:

 - `ne`, `sr`, and `nb` belong to `NA`
- `en`, `ra`, `fi`, etc. belong to `IN`

 All labels compete in the same output space.

 The hierarchy exists only implicitly in the learned parameters.

---

 # Proposed Hierarchical Architecture

 The proposed model introduces an explicit two-stage structure:

```
                 XLM-R Encoder
                       |
                       |
              Shared text representation
                       |
          ------------------------------
          |                            |
          |                            |
 Parent classification            Parent-specific
          |                         experts
          |
   Parent probabilities
          |
   ---------------------
   |    |    |    |    |
  NA   IN   OP   HI   ...
   |    |    |    |
   v    v    v    v

 child expert networks

   |        |        |
 NA child  IN child  OP child
 classifier classifier classifier

   |
   v

 Child predictions
```

 The model has two prediction levels.

---

 # Step 1: Parent Prediction

 First, the shared XLM-R representation is used to predict the parent labels.

 The parent classifier produces probabilities:

 Example:

```
NA = 0.91
IN = 0.87
OP = 0.12
HI = 0.05
```

 These probabilities represent which broad categories are relevant.

 Because this is a multi-label problem, several parents can be active simultaneously.

 For example:

```
Text
 |
 +--> NA
 |
 +--> IN
```

 is possible.

---

 # Step 2: Parent-Specific Experts

 Instead of having one classifier for all child labels, each parent has its own expert.

 Example:

```
NA expert
   |
   +--> ne
   +--> sr
   +--> nb

IN expert
   |
   +--> en
   +--> ra
   +--> dtp
   +--> fi
   +--> lt

OP expert
   |
   +--> rv
   +--> ob
   +--> rs
   +--> av
```

 Each expert specializes in distinguishing only the children belonging to that parent.

 The NA expert does not need to learn about IN labels.

 The IN expert does not need to learn about OP labels.

 This reduces the competition between unrelated labels.

---

 # Training: Soft Differentiable Routing

 During training, the routing is soft.

 Every expert receives the document representation:

```
                 XLM-R
                    |
                    |
              document vector
                    |
       --------------------------------
       |              |               |
       v              v               v

    NA expert      IN expert       OP expert

       |              |               |

   NA logits      IN logits       OP logits

       |              |               |

       x 0.91        x 0.87          x 0.12
```

 The parent probability acts as a weighting factor.

 For example:

```
NA probability = 0.91

NA expert contribution = high

OP probability = 0.12

OP expert contribution = low
```

 The important property is that this is differentiable.

 The child prediction loss can update:

 1. The child classifier
2. The expert
3. The parent classifier

 because the routing weights depend on the parent probabilities.

 Therefore, the model can learn:

 > "If improving child prediction requires changing parent confidence, adjust the parent prediction."

 This allows the hierarchy to be trained jointly.

---

 # Inference: Hard Routing

 During validation and testing, routing becomes discrete.

 The parent probabilities are converted into decisions:

 Example:

```
Parent probabilities:

NA = 0.91
IN = 0.87
OP = 0.12

Threshold = 0.5

Routing:

NA → active
IN → active
OP → inactive
```

 Only active experts are used:

```
             XLM-R

               |
               |

        Parent classifier

               |

        NA = active
        IN = active
        OP = inactive

          |          |

     NA expert   IN expert

          |          |

      children   children
```

 This makes inference more selective because irrelevant experts are ignored.

---

 # Main Differences Compared with Flat Classification

 | Aspect | Flat classifier | Hierarchical expert model |
| --- | --- | --- |
| Label structure | All labels treated equally | Explicit parent-child hierarchy |
| Output layer | One classifier for all labels | Parent classifier + child experts |
| Knowledge of hierarchy | Implicit | Explicit |
| Child prediction | All labels compete together | Children compete only inside their parent |
| Routing | None | Parent predictions control experts |
| Parameter specialization | One shared decision layer | Separate specialized experts |
| Training | Direct label prediction | Joint parent-child optimization |
| Inference | All labels evaluated | Only selected experts evaluated |

---

 # Intuition

 A flat model asks:

 > "Which of these 25 labels apply to this text?"

 The hierarchical model asks:

 > "Which broad categories does this text belong to?"

 and then:

 > "Given those categories, which specific subcategories apply?"

 The assumption is that the second problem is easier because the model does not need to distinguish every possible label relationship at once.

---

 # Why This Architecture Can Help

 The main expected advantages are:

 ### 1\. Reduced label confusion

 Labels that are unrelated do not directly compete.

 For example, the model does not need to decide between:

```
NA child labels

and

OP child labels
```

 inside the same classifier.

---

 ### 2\. Better specialization

 Each expert learns a smaller, more focused classification problem.

 Instead of:

```
25-label decision problem
```

 the model learns several smaller problems:

```
NA expert:
3 labels

IN expert:
5 labels

OP expert:
4 labels
```

---

 ### 3\. Better use of label relationships

 The model explicitly knows:

```
ne belongs to NA
en belongs to IN
rv belongs to OP
```

 rather than discovering these relationships only from training examples.

---

 ### 4\. More efficient inference

 With hard routing:

```
Input
 |
Parent prediction
 |
Only relevant experts activated
 |
Child prediction
```

 The model avoids running every child classifier for every example.

---

 # One-sentence summary

 > I designed a hierarchical multi-label XLM-R architecture where the model first predicts broad parent categories and then uses parent-specific expert classifiers to predict fine-grained child labels. During training, parent probabilities provide differentiable soft routing so the entire hierarchy can be optimized jointly, while during inference, hard routing activates only the experts associated with predicted parents. This contrasts with a flat classifier, where all labels are predicted independently from a single output layer without explicitly using the label hierarchy.