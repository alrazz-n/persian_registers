 # Summary of Hierarchical XLM-R Experiments for Multilabel Classification
 The XLM-R model is the [TurkuNLP's register identification model](https://huggingface.co/TurkuNLP/web-register-classification-multilingual), which in these experiments fine-tuned on the Persian data.

 ## 1\. Research Objective

 The goal of these experiments was to investigate whether explicitly modeling label hierarchy can improve multilingual multilabel text classification.
 Moreover, we want to do error analysis on model's output. To see whether there is a pattern on miss-classification between main-registers or a miss-classification on the main-register that led to not classifiying the sub-resiter.

 The dataset contains **25 labels**, organized into:

 - **9 parent categories** (Main-registers) representing broad semantic groups.
- **16 child categories** (Sub-registers) representing more specific subcategories.

 Unlike ordinary multilabel classification, the labels are not independent. They follow parent-child relationships:

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

IP
 ├── ds
 └── ed

 SP
 ├── it

 ID
 ├── - (no sub registers)

LY
 ├── - (no sub registers)

MT
 ├── - (no sub registers)


```

 A document can belong to multiple parent categories and multiple child categories simultaneously, making the task a **hierarchical multilabel classification problem**.


---

 # 2\. Baseline: Flat Multilabel XLM-R Classifier

 The starting point was a conventional multilabel architecture.

```
             Input text
                  |
                XLM-R
                  |
          Document representation
                  |
          Single classifier
                  |
            25 label outputs
```

 The XLM-R encoder produces a document representation:

 $$
h = XLM-R(x)
$$

 A single classification layer predicts all labels:

 $$
y = sigmoid(Wh+b)
$$

 Each label is treated independently.

 The model learns:

 > "What is the probability of each label given this text?"

 However, the architecture does not explicitly know that:

```
en → IN
fi → IN
ne → NA
rv → OP
```

 The hierarchy can only be learned indirectly from correlations in the training data.

 ### Limitation

 The same representation and classifier must handle:

 - broad categories,
- fine-grained categories,
- unrelated label groups,

 all at the same level.

---

 # 3\. Experiment 1: Multi-Task Hierarchical Classification

 The first hierarchical model introduced separate prediction levels.

 Instead of predicting all labels together, the problem was divided into:

 1. Parent classification
2. Child classification

 Architecture:

```
                    Pretrained XLM-R Encoder
                             |
                             v
                    Shared representation
                       h = X[:, 0, :] (<s>)
                             |
                  -------------------------
                  |                       |
                  v                       v
          Parent classifier       Child classifier
             Linear(H → 9)          Linear(H → 16)
                  |                       |
                  v                       v
             9 parent logits        16 child logits
                  |                       |
                  -----------+-------------
                             |
                             v
                    25 total logits
                    (9 parents + 16 children)


 parent_logits (9)                 child_logits (16)
      |                                  |
      |  vs parent_labels                |  vs child_labels
      v                                  v
  Parent BCE                         Child BCE
      |                                  |
      v                                  v
 parent_weight × parent_loss      child_weight × child_loss
      |                                  |
      +----------------+-----------------+
                       |
                       v
                     loss

Loss = (parent_weight × parent_loss)
     + (child_weight  × child_loss)

```

 The encoder is shared, but two classification objectives are used.

 The total loss is:

 $$
L =
\lambda_pL_{parent}
+
\lambda_cL_{child}
$$

 The parent task teaches the encoder broad semantic information, while the child task learns fine-grained distinctions.

 ### Main contribution

 The hierarchy is introduced through:

 - separate prediction heads,
- hierarchical labels,
- joint optimization.

 Compared with the flat model:

 | Flat model | Hierarchical model |
| --- | --- |
| One prediction task | Two related prediction tasks |
| Labels treated independently | Parent-child relationship modeled |
| One loss | Parent + child loss |

---

 # 4\. Experiment 2: Parent-Conditioned Child Classification

 The next step was to allow parent information to influence child prediction.

 Instead of only sharing the encoder, the child classifier receives a parent-level representation.

 Architecture:

```
                 XLM-R encoder
                       |
              h = <s> hidden state (1024-d) (last_hidden_state[:, 0])
                       |
          -------------------------------
          |                             |
          v                             |
  parent_projection (Linear 1024→256)   |
  + GELU                                |
          |                             |
     parent_hidden (256-d)              |
          |                             |
     ----------------                   |
     |              |                   |
     v              +------> concat <---+
 parent_classifier            [h ; parent_hidden] (1280-d)
 (Linear 256→9)                         |
     |                                  v
 parent_logits                  child_classifier
                                 (Linear 1280→num_children (16))
                                       |
                                  child_logits
                             
 

 parent_logits (9)                 child_logits (16)
      |                                  |
      |  vs parent_labels                |  vs child_labels
      v                                  v
  Parent BCE                         Child BCE
      |                                  |
      v                                  v
 parent_weight × parent_loss      child_weight × child_loss
      |                                  |
      +----------------+-----------------+
                       |
                       v
                     loss


(For joint output / evaluation / prediction only)
all_logits = [parent_logits ; child_logits]  (25-d)

```

 A parent representation is learned:

 $$
h_p = GELU(W_ph+b)
$$

 The child classifier receives:

 $$
[h;h_p]
$$

 Therefore, child prediction uses:

 - the original document representation,
- the learned parent-level information.

 ### Motivation

 The model no longer only asks:

 > "What child label is present?"

 It also considers:

 > "What broad category does this text belong to?"

 This creates a softer hierarchy because the child classifier receives parent information without depending on hard parent predictions.

---

 # 5\. Experiment 3: Parent-Specific Expert Networks

 The next improvement introduced specialization.

 Instead of one child classifier:

```
One child classifier
        |
 all child labels
```

 separate experts were created:

```
                        XLM-R encoder
                             |
                             v
                    h  <s> hidden state (1024-d)
                       (last_hidden_state[:, 0])
                             |
        ---------------------|-----------------------
        |                                           |
        v                                           v
 PARENT BRANCH                         PARENT-SPECIFIC EXPERTS
 Linear 1024→256 + GELU                (all 6 run on every sample, no routing,
        |                                each reads the same h)
        |
        |                                inside each expert (separate weights):
        |                                h [1024] → Linear 1024→256 → GELU
        |                                → Dropout(0.1) → Linear 256→256 → GELU
        |                                → [256-d expert hidden]
        |                                → Child classifier (256→k) → k child logits
        |
 Linear 256→9
 (parent classifier)                                  |
        |                              ┌────────┬───────────────────────┬─────────────────┐
 [9 parent logits]                     │ Expert │ Child classifier      │ Output          │
 MT LY SP ID NA HI IN OP IP            ├────────┼───────────────────────┼─────────────────┤
        |                              │ SP     │ 256→1                 │ it              │
        |                              │ NA     │ 256→3                 │ ne sr nb       │
        |                              │ HI     │ 256→1                 │ re              │
        |                              │ IN     │ 256→5                 │ en ra dtp fi lt │
        |                              │ OP     │ 256→4                 │ rv ob rs av     │
        |                              │ IP     │ 256→2                 │ ds ed           │
        |                              └────────┴───────────────────────┴─────────────────┘
        |                                               |
        |                                          concat → [16 child logits]
        |                                               |
        ------------------------------------------------
                             |
                             v
                  Concatenate [9 parent | 16 child]
                             |
                             v
                          [25 logits]





Parent_logits (9)                 child_logits (16)
      |                                  |
      |  vs parent_labels                |  vs child_labels
      v                                  v
  Parent BCE                         Child BCE
      |                                  |
      v                                  v
 parent_weight × parent_loss      child_weight × child_loss
      |                                  |
      +----------------+-----------------+
                       |
                       v
                     loss
```

 Each expert specializes in one parent category.

 For example:

 ### IN expert

 Only learns:

```
en
ra
dtp
fi
lt
```

 ### NA expert

 Only learns:

```
ne
sr
nb
```

 The motivation is that different label groups may require different feature transformations.

 Instead of one classifier learning everything, the model learns several smaller problems.

---

 # 6\. Experiment 4 and 5: Hard-Routed Hierarchical Experts

 The next stage introduced explicit routing.

 The model first predicts parents:

```
Parent classifier

NA = 0.91
IN = 0.87
OP = 0.12
```

 Then activates only relevant experts:

```
NA active → NA expert
IN active → IN expert
OP inactive → skipped
```

 The architecture becomes:

```
                 XLM-R

                    |

            Parent classifier

                    |

              Parent routing

                    |

        ----------------------
        |                    |

     NA expert            IN expert

        |                    |

    NA children        IN children
```

 ## Training

 To avoid early parent errors preventing child learning, training uses gold parent labels.

 Example:

```
True parents:

NA = 1
IN = 1
OP = 0
```

 The correct experts receive training signals.

 ## Inference

 At test time:

```
Predicted parents → routing → experts
```

 This creates a realistic hierarchical pipeline.

 ### Advantage

 The model reduces competition between unrelated labels.

 ### Limitation

 Routing errors can propagate:

```
Wrong parent prediction
        |
        |
Child expert not activated
        |
        |
Child prediction fails
```

---

 # 7\. Experiment 6: Soft Training Routing and Hard Inference Routing

 The next architecture combined the benefits of differentiability and specialization.

 Instead of selecting experts using hard decisions during training, parent probabilities were used as soft weights.

 Example:

```
Parent probabilities:

IN = 0.8
SP = 0.7
NA = 0.1
```

 Experts produce representations:

 $$
E_{IN}(h)
$$

 $$
E_{SP}(h)
$$

 $$
E_{NA}(h)
$$

 The final representation becomes:

 $$
h_{mix}
=
\sum_p w_pE_p(h)
$$

 where $w_p$ represents the parent probability.

 This allows gradients to flow through the routing mechanism.

 Therefore:

 - child loss updates experts,
- child loss updates the parent router,
- the complete hierarchy is optimized jointly.

 During inference, routing becomes hard:

```
IN = active
SP = active
NA = inactive
```

 Only selected experts contribute.

---

 # 8\. Experiment 7: Parent-Specific Mixture-of-Experts

 The final architecture refined the expert idea into a full hierarchical mixture-of-experts model.

 The architecture:

```
                 XLM-R

                    |

            Parent classifier

                    |

              Routing weights

                    |

      --------------------------------

      SP expert   NA expert   IN expert

                    |

            Weighted combination

                    |

          Shared child classifier
```

 Unlike hard routing, the model does not completely select one expert.

 Instead, multiple experts contribute according to parent probabilities.

 The child representation is:

 $$
h_{mix}
=
\sum_p w_pE_p(h)
$$

 The child classifier receives this parent-aware representation:

 $$
child\_logits=W_ch_{mix}+b
$$

 ### Main idea

 Different parent categories may require different feature transformations.

 Therefore:

 - the encoder learns general linguistic information,
- experts learn parent-specific information,
- the router determines which expertise is useful.

---

 # 9\. Evolution of the Models

 The experiments represent a gradual increase in hierarchical modeling:

```
Flat classifier
      |
      v
Separate parent and child tasks
      |
      v
Parent-conditioned child prediction
      |
      v
Parent-specific experts
      |
      v
Hard expert routing
      |
      v
Soft routing during training
      |
      v
Mixture-of-experts hierarchy
```

 Each experiment adds more explicit use of the label structure.

---

 # 10\. Overall Comparison

 | Model | Main Idea | Hierarchy Usage |
| --- | --- | --- |
| Flat XLM-R | Predict all labels independently | None |
| Hierarchical multitask | Separate parent and child objectives | Loss-level hierarchy |
| Parent-conditioned model | Child receives parent representation | Feature-level hierarchy |
| Expert model | Separate classifiers per parent | Parameter specialization |
| Hard routing model | Activate only relevant experts | Conditional computation |
| Soft routing model | Parent probabilities weight experts | Differentiable hierarchy |
| Hierarchical MoE | Learn weighted expert mixtures | Full hierarchical representation learning |

---

 # 11\. Overall Research Contribution

 Across these experiments, the model design evolved from a standard multilabel classifier into a hierarchical architecture that explicitly incorporates label relationships.

 The main contributions are:

 1. **Using label hierarchy as additional knowledge**

 Instead of treating labels as unrelated outputs, the model uses known parent-child relationships.

 2. **Separating general and fine-grained information**

 Parent classifiers learn broad semantic categories, while child classifiers focus on detailed distinctions.

 3. **Introducing specialized representations**

 Parent-specific experts allow different label groups to learn different transformations.

 4. **Exploring different routing strategies**

 The experiments compare:

 - no routing,
- hard routing,
- soft routing,
- mixture-of-experts routing.

 5. **Maintaining end-to-end optimization**

 The final models allow parent and child objectives to jointly improve the shared XLM-R representation.

---

 # Short Presentation Summary

 > I investigated several hierarchical XLM-R architectures for multilabel classification. Starting from a flat classifier where all labels are predicted independently, I gradually introduced label hierarchy through multi-task learning, parent-conditioned prediction, parent-specific experts, and routing mechanisms. The final models use parent predictions not only as outputs but also as information for constructing specialized child representations. The experiments explore different ways of exploiting hierarchy, ranging from explicit parent-child losses to hard and soft expert routing. The overall objective is to allow the model to learn both global category structure and fine-grained label distinctions while reducing competition between unrelated labels.




 | Model | Parent Prediction | Child Prediction | Expert Networks | Routing Strategy | Hierarchy Usage |
| --- | --- | --- | --- | --- | --- |
| Flat XLM-R baseline | No | Single classifier predicts all labels | No | None | No explicit hierarchy |
| Hierarchical multitask XLM-R (Exp. 1) | Separate parent head | Separate child head | No | None | Parent and child learned with separate losses |
| Parent-conditioned XLM-R (Exp. 2) | Parent representation learned | Child classifier receives parent representation | No | Soft feature conditioning | Parent information influences child prediction |
| Parent-specific experts (Exp. 3) | Parent classifier | Separate child expert per parent | Yes | No routing | Each parent group learns specialized transformations |
| Hard-routed experts (Exp. 4/5) | Parent classifier decides active groups | Only selected experts predict children | Yes | Hard routing | Explicit conditional computation |
| Soft-routing experts (Exp. 6) | Parent probabilities act as weights | Weighted expert representations | Yes | Differentiable soft routing | Parent confidence controls expert contribution |
| Hierarchical MoE (Exp. 7) | Parent classifier acts as router | Shared child classifier uses mixed expert representation | Yes | Mixture-of-experts | Full parent-aware representation learning |

---

 | Model | Main Strength | Main Weakness |
| --- | --- | --- |
| Flat classifier | Simple, stable, no routing errors | Ignores label hierarchy |
| Multitask hierarchy | Easy way to introduce hierarchy | Child classifier does not directly use parents |
| Parent-conditioned model | Uses parent information while remaining differentiable | Still has shared child classifier |
| Expert model | Allows specialization | More parameters |
| Hard routing | Efficient and interpretable | Parent errors propagate to children |
| Soft routing | Fully trainable end-to-end | More computational cost |
| Hierarchical MoE | Flexible parent-aware representations | More complex architecture |


 ### 4\. Results table (if you have metrics)

| Model                                  | Micro-F1 | Macro-F1 | Parent Micro-F1 | Parent Macro-F1 | Child Micro-F1 | Child Macro-F1 |
|----------------------------------------|----------|----------|-----------------|-----------------|----------------|----------------|
| Flat XLM-R                             | 0.77     | 0.76     | 0.79               | 0.77               | 0.73              | 0.75              |
| Hierarchical Multitask XLM-R           | 0.77     | 0.77     | 0.80            | 0.78            | 0.73           | 0.76           |
| Parent-Conditioned Hierarchical XLM-R  | 0.76     | 0.75     | 0.78               | 0.76               | 0.73              |  0.74             |
| Parent-Specific Expert XLM-R            | 0.76     | 0.74     | 0.80            | 0.79            | 0.69           | 0.72           |
| GoldRoute Hierarchical Expert XLM-R     | 0.76        | 0.75        | 0.78               | 0.76               | 0.73              | 0.74              |
| HardRoute Hierarchical Expert XLM-R     | 0.73     | 0.70     | 0.75            | 0.69            | 0.70           | 0.71           |
| SoftRoute Hierarchical Expert XLM-R     | 0.73     | 0.66     | 0.80            | 0.79            | 0.60           | 0.58           |
| Hierarchical Mixture-of-Experts XLM-R   | 0.35     | 0.21     | 0.61            | 0.39            | 0.13           | 0.11           |


Baseline:

    - Flat XLM-R

Hierarchy-aware models:

    - Exp.1  Hierarchical Multitask XLM-R
    - Exp.2  Parent-Conditioned Hierarchical XLM-R

Expert-based models:

    - Exp.3  Parent-Specific Expert XLM-R
    - Exp.4  Hard-Routed Expert XLM-R
    - Exp.5  Parent-Guided Hard Routing XLM-R
    - Exp.6  Soft-Routed Expert XLM-R
    - Exp.7  Hierarchical Mixture-of-Experts XLM-R

| Experiment | Suggested Model Name | Short Description |
|---|---|---|
| Experiment 1 | Hierarchical Multitask XLM-R | Separate parent and child classification heads trained jointly with a weighted parent-child loss. |
| Experiment 2 | Parent-Conditioned Hierarchical XLM-R | Child classifier receives a learned parent representation in addition to the original XLM-R representation. |
| Experiment 3 | Hierarchical Parent-Specific Expert XLM-R | Separate expert networks are created for each parent category, with each expert predicting its own child labels. |
| Experiment 4 | Hierarchical Hard-Routed Expert XLM-R (GoldRoute) | Parent predictions determine which child experts are activated. Uses hard routing. |
| Experiment 5 | Hierarchical Parent-Based Hard Routing XLM-R | Same general idea as Experiment 4, emphasizing predicted parent routing and parent-child pipeline behavior. |
| Experiment 6 | Hierarchical Soft-Routed Expert XLM-R | Parent probabilities are used as differentiable routing weights during training and hard routing during inference. |
| Experiment 7 | Hierarchical Parent-Specific Mixture-of-Experts XLM-R (Hierarchical MoE XLM-R) | Parent probabilities create a weighted mixture of expert representations before child classification. |


| Exp. | Paper-style Name |
|---|---|
| Baseline | Flat XLM-R Multilabel Classifier |
| 1 | Hierarchical Multitask XLM-R (HMT-XLMR) |
| 2 | Parent-Conditioned Hierarchical XLM-R (PC-H-XLMR) |
| 3 | Parent-Specific Expert Hierarchical XLM-R (PSE-H-XLMR) |
| 4 | Hard-Routed Hierarchical Expert XLM-R (HR-H-XLMR) |
| 5 | Parent-Guided Hard Routing XLM-R (PG-HR-XLMR) |
| 6 | Soft-Routed Hierarchical Expert XLM-R (SR-H-XLMR) |
| 7 | Hierarchical Mixture-of-Experts XLM-R (H-MoE-XLMR) |

| ID       | Model                                      | Description                                      | Results row            |
|----------|--------------------------------------------|--------------------------------------------------|------------------------|
| Baseline | Flat XLM-R                                  | Standard multilabel classifier                  | Flat                   |
| Exp. 1   | Hierarchical Multitask XLM-R              | Separate parent and child heads with joint loss | Hierarchical multitask |
| Exp. 2   | Parent-Conditioned Hierarchical XLM-R     | Child classifier receives parent representation  | Parent-conditioned     |
| Exp. 3   | Parent-Specific Expert XLM-R              | Separate child experts per parent               | Expert model           |
| Exp. 4   | GoldRoute Hierarchical Expert XLM-R       | Training uses gold parent labels for routing    | ---                |
| Exp. 5   | HardRoute Hierarchical Expert XLM-R       | Routing uses predicted parent labels             | Hard routing           |
| Exp. 6   | SoftRoute Hierarchical Expert XLM-R       | Parent probabilities softly weight experts      | Soft MoE               |
| Exp. 7   | Hierarchical Mixture-of-Experts XLM-R     | Full MoE representation learning                | Hierarchical MoE       |

----------------------------------

# GoldRoute Vs HardRoute Vs SoftRoute (Exp 4, 5 , 6)

 The core architecture is similar:

```
                 XLM-R
                    |
                    |
            Parent classifier
                    |
                    |
            Parent information
                    |
                    |
          Parent-specific experts
                    |
                    |
            Child predictions
```

 The difference is the **routing mechanism**.

 A good way to explain them is:

---

 # Experiment 4: GoldRoute — Gold Parent Routing

 ## Model name

 **Hierarchical Gold-Routed Expert XLM-R**

 (or)

 **Gold-Label Routed Hierarchical Expert Model**

---

 ## Main idea

 During training, the model does **not use its own parent predictions** to decide which experts are activated.

 Instead, it uses the **ground-truth parent labels** from the dataset.

 The routing information is perfect because it comes from the annotations.

 Example:

 The training label is:

```
Parent labels:

NA = 1
IN = 1
OP = 0
```

 The routing is:

```
                 XLM-R

                    |

            Parent labels (gold)

                    |

        -------------------------
        |                       |

        v                       v

    NA expert              IN expert

        |                       |

    ne sr nb             en ra dtp fi lt
```

 The child experts receive only examples that belong to their parent.

---

 ## Why use GoldRoute?

 The main motivation is to avoid the **cold-start problem**.

 At the beginning of training, the parent classifier is not accurate.

 If routing depended on predicted parents:

```
Wrong parent prediction

        ↓

Wrong expert activated

        ↓

Correct child expert receives no training examples
```

 The child experts would have difficulty learning.

 Gold routing solves this:

```
Correct parent known

        ↓

Correct expert always receives training data

        ↓

Expert learns its child categories
```

---

 ## Advantage

 - Stable training.
- Every child expert receives correct examples.
- No routing errors during training.

---

 ## Disadvantage

 There is a mismatch:

 Training:

```
Gold parents → experts
```

 Testing:

```
Predicted parents → experts
```

 The model never learns how to handle incorrect routing.

 This is called a **train-test routing mismatch**.

---

 # Experiment 5: HardRoute — Predicted Hard Routing

 ## Model name

 **Hierarchical Hard-Routed Expert XLM-R**

---

 ## Main idea

 Unlike GoldRoute, the model uses its own parent predictions to decide which experts are activated.

 The parent classifier becomes a router.

 The process is:

```
Text

 ↓

XLM-R

 ↓

Parent classifier

 ↓

Parent probabilities

 ↓

Threshold

 ↓

Active experts

 ↓

Child prediction
```

 Example:

 Parent predictions:

```
NA = 0.91
IN = 0.83
OP = 0.12
```

 Threshold = 0.5

 Routing:

```
NA → active
IN → active
OP → inactive
```

 Only:

```
NA expert
IN expert
```

 are used.

---

 ## Difference from GoldRoute

 The difference is **where the routing decision comes from**.

 |  | GoldRoute | HardRoute |
| --- | --- | --- |
| Routing source | True labels | Model predictions |
| Training routing | Perfect | Imperfect |
| Inference routing | Predicted | Predicted |
| Train-test mismatch | Yes | No |
| Risk of routing error | No during training | Yes |

---

 ## Advantage

 The training process matches inference.

 The model learns the real deployment behavior:

```
Predict parents → select experts → predict children
```

---

 ## Disadvantage

 Errors propagate.

 Example:

 The document actually contains:

```
IN = true
en = true
```

 but the parent classifier predicts:

```
IN = false
```

 Then:

```
IN expert is not activated

        ↓

en cannot be predicted
```

 This is called **hierarchical error propagation**.

---

 # Experiment 6: SoftRoute — Differentiable Soft Routing

 ## Model name

 **Hierarchical Soft-Routed Expert XLM-R**

 (or)

 **Hierarchical Mixture Routing XLM-R**

---

 ## Main idea

 SoftRoute avoids the binary decision:

```
Expert activated / not activated
```

 Instead, every expert contributes according to the parent probability.

 Example:

 Parent predictions:

```
IN = 0.80
SP = 0.70
NA = 0.10
```

 Instead of:

```
IN expert → yes
SP expert → yes
NA expert → no
```

 the model uses:

```
IN expert contribution = 0.80

SP expert contribution = 0.70

NA expert contribution = 0.10
```

 The expert representations are combined:

 $$
h_{mix}
=
0.8E_{IN}(h)
+
0.7E_{SP}(h)
+
0.1E_{NA}(h)
$$

 The child classifier receives:

```
Parent-aware mixed representation
```

---

 ## Why SoftRoute?

 The problem with HardRoute:

```
Parent prediction
       |
       |
binary decision
       |
       |
expert selected
```

 The threshold operation is not differentiable.

 For example:

```
IN = 0.49

IN expert OFF
```

 but:

```
IN = 0.51

IN expert ON
```

 A tiny probability change causes a large architectural change.

 Soft routing avoids this.

 Instead:

```
IN = 0.49

IN expert contributes a little
```

---

 ## Advantage

 The whole model is trainable end-to-end.

 The child loss can improve the router:

 Example:

 If child prediction improves when IN expert contributes more:

```
Child loss
      |
      ↓
IN expert
      |
      ↓
Parent probability
```

 The model learns better routing automatically.

---

 ## Disadvantage

 - More computational cost because several experts contribute.
- Experts are not completely isolated.
- Routing is less interpretable than hard selection.

---

 # Direct Comparison: GoldRoute vs HardRoute vs SoftRoute

 | Feature | GoldRoute (Exp. 4) | HardRoute (Exp. 5) | SoftRoute (Exp. 6) |
| --- | --- | --- | --- |
| Routing information | Ground-truth parents | Predicted parents | Predicted probabilities |
| Routing type | Perfect hard routing | Binary hard routing | Continuous weighted routing |
| Expert activation | Based on labels | Based on threshold | Based on probability |
| Differentiable | Yes (child model), but routing is external | No | Yes |
| Training stability | Highest | Lower | High |
| Matches inference | No | Yes | Yes |
| Routing errors | No during training | Possible | Reduced |
| Computational cost | Low | Lowest | Higher |
| Main purpose | Train experts correctly | Realistic hierarchical pipeline | Learn adaptive routing |

---

 # How to Explain the Evolution

 A very clear explanation for a thesis would be:

 > The GoldRoute model was introduced to study the effect of ideal hierarchical routing by using ground-truth parent labels during training. This ensures that each child expert receives the correct training examples, but creates a mismatch between training and inference. The HardRoute model removes this mismatch by using predicted parent labels as routing decisions, making the model behave like a real hierarchical classifier. However, hard routing introduces error propagation because incorrect parent predictions can prevent child experts from being activated. The SoftRoute model addresses this limitation by replacing discrete routing with differentiable probability-based routing. Instead of selecting experts completely, parent probabilities determine their contribution, allowing the entire hierarchy to be optimized jointly.

---

 In your experiment table, I would therefore rename them as:

 | Experiment | Name |
| --- | --- |
| Exp. 4 | **GoldRoute: Hierarchical Gold-Routed Expert XLM-R** |
| Exp. 5 | **HardRoute: Hierarchical Hard-Routed Expert XLM-R** |
| Exp. 6 | **SoftRoute: Hierarchical Soft-Routed Expert XLM-R** |

---

With this corrected table, the interpretation becomes more interesting because now you can separate **parent-level performance**, **child-level performance**, and **overall performance**. The results suggest that the main difficulty is not learning parent categories; it is using the hierarchy effectively for fine-grained child prediction.

 A possible interpretation section for a thesis/paper could look like this:

---

 # Results Interpretation

 The experimental results show that explicitly incorporating label hierarchy does not automatically improve multilabel classification performance. The flat XLM-R model achieves a Micro-F1 of **0.77** and Macro-F1 of **0.76**, providing a strong baseline. Several hierarchical approaches achieve comparable performance, indicating that the encoder already captures some parent-child relationships implicitly. However, models that introduce stronger hierarchical constraints, such as routing and mixture-of-experts mechanisms, face additional optimization challenges.

---

 ## Flat XLM-R Baseline

 The flat XLM-R model achieves:

 - Micro-F1: **0.77**
- Macro-F1: **0.76**

 with:

 - Parent Micro-F1: **0.78**
- Child Micro-F1: **0.66**

 The difference between parent and child performance is important.

 The model performs substantially better on parent labels than child labels:

```
Parent Micro-F1: 0.78
Child Micro-F1: 0.66
```

 This indicates that the broad categories are easier to identify, while the fine-grained child categories are more challenging.

 This is expected because parent labels represent broader semantic concepts, whereas child labels require distinguishing between closely related categories.

 For example:

```
IN
 ├── en
 ├── ra
 ├── dtp
 ├── fi
 └── lt
```

 Once the model recognizes the general category `IN`, it still needs to distinguish between several similar subcategories.

 This suggests that the main difficulty of the task lies in **fine-grained classification rather than parent recognition**.

---

 # Hierarchical Multitask XLM-R

 The hierarchical multitask model achieves the same overall performance as the baseline:

 |  | Flat | Hierarchical Multitask |
| --- | --- | --- |
| Micro-F1 | 0.77 | 0.77 |
| Macro-F1 | 0.76 | 0.76 |

However, the internal performance improves:

 |  | Parent | Child |
| --- | --- | --- |
| Parent Micro-F1 | 0.78 | **0.80** |
| Child Micro-F1 | 0.66 | **0.73** |

The improvement in child performance suggests that the additional parent classification objective provides a useful learning signal.

 The shared encoder receives gradients from two related tasks:

```
             XLM-R

              |
      -----------------
      |               |

 Parent task      Child task
```

 The parent objective encourages the encoder to learn broad semantic information, while the child objective maintains fine-grained discrimination.

 The fact that the overall Micro-F1 does not increase despite better child performance suggests that improvements in some labels may be balanced by decreases or unchanged performance in others.

 This result indicates that:

 > Adding hierarchical supervision is beneficial, but simply adding a second classification objective is not sufficient to produce large improvements over a strong flat baseline.

---

 # Parent-Conditioned Hierarchical XLM-R

 The parent-conditioned model performs worse:

 - Micro-F1 decreases from **0.77 → 0.74**
- Macro-F1 decreases from **0.76 → 0.71**

 Although child performance remains relatively high:

```
Child Micro-F1 = 0.71
Child Macro-F1 = 0.73
```

 the model does not benefit from explicitly injecting parent representations into child prediction.

 A possible explanation is that parent information may be too coarse for fine-grained classification.

 The parent representation learns:

```
"This document belongs to IN"
```

 but the child classifier needs:

```
"Which IN subtype is this?"
```

 Therefore, the parent representation may not contain enough discriminative information for separating child categories.

 This suggests that the relationship between parent and child labels is not necessarily a simple predictive dependency. Knowing the parent category reduces the search space, but does not necessarily provide the features required for child discrimination.

---

 # Parent-Specific Expert XLM-R

 The expert model introduces specialization by assigning different expert networks to different parent groups.

 The performance is:

 - Micro-F1: **0.73**
- Macro-F1: **0.63**

 Although the overall performance decreases, parent performance remains strong:

```
Parent Micro-F1 = 0.80
Parent Macro-F1 = 0.78
```

 The child performance:

```
Child Micro-F1 = 0.71
Child Macro-F1 = 0.73
```

 is comparable to the parent-conditioned model.

 This suggests that creating separate experts is not inherently beneficial or harmful. The model can learn parent-specific transformations, but splitting the model into experts may reduce the amount of shared information available for learning rare child categories.

 A possible explanation is data fragmentation:

 Flat model:

```
All examples
     |
All labels
```

 Expert model:

```
IN examples → IN expert

NA examples → NA expert

OP examples → OP expert
```

 Each expert receives fewer training examples, which can make learning difficult, especially for rare child labels.

---

 # GoldRoute Hierarchical Expert XLM-R

 GoldRoute uses true parent labels for routing during training.

 The model achieves:

 - Micro-F1: **0.76**
- Macro-F1: **0.75**

 which is close to the flat baseline.

 This is an important result because it shows that the expert architecture itself is capable of learning useful representations when routing is reliable.

 However, child performance is lower:

```
Child Micro-F1 = 0.59
Child Macro-F1 = 0.54
```

 Compared with the hierarchical multitask model:

```
Hierarchical multitask child Micro-F1 = 0.73
GoldRoute child Micro-F1 = 0.59
```

 This suggests that hard separation of child classifiers may be too restrictive.

 Even with perfect routing, experts may lose useful cross-category information because each expert only sees its own parent group.

 Therefore:

 > The hierarchy provides useful information, but completely separating the label space can remove beneficial parameter sharing.

---

 # HardRoute Hierarchical Expert XLM-R

 HardRoute replaces gold routing with predicted parent routing.

 Performance decreases compared with GoldRoute:

 |  | GoldRoute | HardRoute |
| --- | --- | --- |
| Micro-F1 | 0.76 | 0.73 |
| Macro-F1 | 0.75 | 0.70 |

This indicates the effect of routing errors.

 The prediction process becomes:

```
Parent prediction
        |
        |
Expert selection
        |
        |
Child prediction
```

 An incorrect parent prediction can prevent the correct child expert from being activated.

 For example:

```
True:

IN = 1
en = 1

Prediction:

IN = 0
```

 The model cannot reach the IN expert, causing the child prediction to fail.

 The decrease from GoldRoute to HardRoute demonstrates that parent prediction accuracy is a bottleneck for hierarchical routing models.

---

 # SoftRoute Hierarchical Expert XLM-R

 SoftRoute uses parent probabilities instead of binary routing decisions.

 The parent performance is strong:

```
Parent Micro-F1 = 0.80
Parent Macro-F1 = 0.79
```

 However, child performance drops:

```
Child Micro-F1 = 0.60
Child Macro-F1 = 0.58
```

 This suggests that the issue is not parent classification.

 The model successfully learns parent categories but struggles to transform parent probabilities into useful child representations.

 Possible explanations include:

 ### 1\. Weak routing signal

 A parent probability indicates confidence in a category but does not necessarily represent which expert transformation is needed.

 For example:

```
IN = 0.85
```

 does not specify whether the useful expert information is related to:

```
en
ra
dtp
fi
lt
```

 ### 2\. Expert competition

 Multiple experts contribute simultaneously:

 $$
h_{mix}=\sum_p w_pE_p(h)
$$

 If many experts contribute, the child classifier may receive a noisy mixed representation.

---

 # Hierarchical Mixture-of-Experts XLM-R

 The final MoE model shows a significant performance collapse:

 - Micro-F1: **0.35**
- Macro-F1: **0.21**

 The parent and child performance also decrease:

```
Parent Micro-F1 = 0.61

Child Micro-F1 = 0.13
```

 This indicates that the model failed to learn a stable routing mechanism and expert representation.

 Compared with SoftRoute:

```
SoftRoute:
Parent F1 = 0.80
Child F1 = 0.60

Hierarchical MoE:
Parent F1 = 0.61
Child F1 = 0.13
```

 the failure occurs earlier in the pipeline.

 Possible reasons include:

 - increased optimization complexity,
- insufficient training data for expert specialization,
- unstable interaction between router, experts, and classifier,
- mismatch between the assumed hierarchy and the true label structure.

---

 # Overall Findings

 The experiments reveal several important conclusions.

 ## 1\. Parent labels are easier than child labels

 Across models:

```
Parent F1 > Child F1
```

 This confirms that fine-grained classification is the main challenge.

---

 ## 2\. Hierarchical supervision helps, but strong structural constraints hurt

 The best hierarchical improvement comes from the multitask model:

```
Hierarchical Multitask:
Child Micro-F1 = 0.73
```

 This suggests that the hierarchy is useful as additional supervision.

 However, forcing predictions through routing mechanisms reduces performance.

---

 ## 3\. Parameter sharing remains important

 The flat model and hierarchical multitask model perform similarly or better than expert models.

 This suggests that sharing information between categories is important, especially when child labels are rare.

---

 ## 4\. Routing accuracy is a critical bottleneck

 The comparison:

```
GoldRoute → HardRoute
0.76 → 0.73 Micro-F1
```

 shows that predicted routing introduces additional errors.

---

 ## 5\. The annotation hierarchy may represent correlation rather than strict dependency

 The results suggest that the hierarchy is useful, but not necessarily strong enough to support a strict decision process:

```
parent → expert → child
```

 The data may contain:

 - overlapping categories,
- incomplete parent-child relationships,
- children that require information beyond their parent category.

 Therefore, models that use hierarchy as an auxiliary signal perform better than models that enforce hierarchy as a hard computational constraint.

---

 This interpretation gives you a strong discussion section because it explains **why the simpler hierarchical model works while the more complex routing models fail**. It frames the results as an investigation of the limits of hierarchical modeling rather than just a list of scores.



---

 # Summary of Experiments (Supervisor mode)

 I investigated whether explicitly modeling the parent-child label hierarchy improves multilabel text classification with XLM-R.

 The baseline model was a **flat XLM-R multilabel classifier**, where all labels were predicted independently.

 Then I introduced several hierarchical approaches:

 | Model | Main idea |
| --- | --- |
| **Flat XLM-R** | Baseline. All labels predicted independently from one shared representation. |
| **Hierarchical Multitask XLM-R** | Separate parent and child classifiers trained jointly. The encoder learns from both coarse and fine-grained objectives. |
| **Parent-Conditioned Hierarchical XLM-R** | Child prediction receives additional parent-level representation to incorporate hierarchical information. |
| **Parent-Specific Expert XLM-R** | Separate expert networks are created for different parent categories, allowing specialization. |
| **GoldRoute Hierarchical Expert XLM-R** | Child experts are selected using true parent labels during training to study the effect of perfect routing. |
| **HardRoute Hierarchical Expert XLM-R** | Child experts are selected using predicted parent labels, simulating real hierarchical inference. |
| **SoftRoute Hierarchical Expert XLM-R** | Parent probabilities are used as soft routing weights, allowing differentiable expert selection. |
| **Hierarchical Mixture-of-Experts XLM-R** | Parent predictions create a weighted mixture of expert representations before child classification. |

The experiments gradually increased the amount of hierarchical structure imposed on the model:

```
Flat prediction
        ↓
Hierarchy as additional supervision
        ↓
Hierarchy as conditioning information
        ↓
Hierarchy as expert specialization
        ↓
Hierarchy as routing mechanism
        ↓
Hierarchy as mixture-of-experts computation
```

---

 # Main Findings From Results

 ## 1\. Hierarchy helps when used as supervision, not as strict structure

 The strongest hierarchical model was:

 **Hierarchical Multitask XLM-R**

 - Overall Micro-F1: 0.77 (same as flat baseline)
- Child Micro-F1: 0.73 vs 0.66 for flat model

 This suggests that the parent-child hierarchy contains useful information, especially for improving child classification.

 However, forcing the model to strictly follow:

```
parent → expert → child
```

 generally reduced performance.

---

 ## 2\. Parent prediction is easier than child prediction

 Across almost all models:

```
Parent F1 > Child F1
```

 Example:

 Flat XLM-R:

```
Parent Micro-F1: 0.78
Child Micro-F1: 0.66
```

 This indicates that the main difficulty is not recognizing broad categories, but distinguishing between fine-grained child labels.

 The dataset appears to contain relatively clear high-level categories but more ambiguous subcategories.

---

 # Possible Model Optimization Issues

 ## 1\. Expert models may suffer from reduced parameter sharing

 The expert-based models divide the learning problem:

 Before:

```
One model learns all labels together
```

 After:

```
IN expert → IN children
NA expert → NA children
OP expert → OP children
```

 This specialization can be useful, but it also reduces the amount of training data each expert receives.

 Possible consequence:

 - Rare child labels have fewer examples.
- Experts may overfit.
- Shared linguistic information between categories is lost.

 This may explain why expert models did not outperform the multitask model.

---

 ## 2\. Routing introduces error propagation

 Hard routing creates a dependency chain:

```
Parent prediction
        ↓
Expert selection
        ↓
Child prediction
```

 If the parent prediction is wrong, the correct child classifier may never be activated.

 The difference between:

```
GoldRoute:
Micro-F1 = 0.76

HardRoute:
Micro-F1 = 0.73
```

 shows that routing errors directly affect child prediction.

---

 ## 3\. More complex models are harder to optimize

 The hierarchical MoE model performs poorly:

```
Micro-F1 = 0.35
```

 This suggests optimization difficulties rather than only architectural limitations.

 Possible causes:

 - Too many interacting components (router + experts + classifier).
- Difficult gradient flow.
- Insufficient data for learning expert specialization.
- Instability between routing decisions and expert learning.

 The results suggest that increasing architectural complexity does not automatically improve performance.

---

 # Possible Annotation Schema Issues

 ## 1\. The hierarchy may represent correlation rather than strict dependency

 The experiments suggest that:

```
child → parent
```

 is useful information, but it may not be a strict decision process.

 A strict hierarchical model assumes:

```
No parent → no child
```

 However, the dataset may contain cases where:

 - children contain information not captured by parents,
- parent-child relationships are incomplete,
- multiple categories overlap.

 This would explain why soft hierarchical supervision works better than hard routing.

---

 ## 2\. Parent categories may be too broad

 Parent labels are relatively easy:

```
Parent F1 ≈ 0.80
```

 but children are harder:

```
Child F1 ≈ 0.70
```

 This suggests that parent categories provide limited information for separating children.

 For example:

```
IN
 ├── en
 ├── ra
 ├── dtp
 ├── fi
 └── lt
```

 The parent `IN` tells the model the general domain, but not necessarily which subtype is correct.

---

 ## 3\. Child labels may have overlapping semantics

 The difficulty of child prediction suggests that some child categories may not have clear boundaries.

 Possible issues:

 - Similar linguistic patterns between children.
- Ambiguous annotation decisions.
- Multiple valid interpretations of the same text.

 This could make expert specialization less effective because experts assume that child categories inside a parent form a clearly separable group.

---

 # Overall Conclusion for Supervisor

 > The experiments show that the label hierarchy contains useful information, but the most effective way to use it is as additional supervision rather than as a strict routing mechanism. The hierarchical multitask model improved child classification by allowing parent and child objectives to jointly train the encoder. In contrast, expert and routing-based models introduced optimization difficulties and error propagation. The results suggest that the annotation hierarchy captures meaningful high-level structure, but it may not represent a strict decision tree. Child labels appear to be more ambiguous and difficult to separate, which limits the effectiveness of hard hierarchical architectures. Future improvements may focus on softer hierarchical representations, better label modeling, or investigating the consistency of parent-child annotations.
