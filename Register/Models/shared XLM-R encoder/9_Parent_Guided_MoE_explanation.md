 # Hierarchical XLM-R with Parent-Specific Mixture-of-Experts

 ## Motivation

 The task is a **hierarchical multi-label classification problem**. The labels are not independent; they have a parent-child relationship.

 For example:

 - A text can belong to the parent category **SP**
- Under SP, it can have the child label **it**

 or:

 - A text can belong to **IN**
- Under IN, it can have one or more children such as **en**, **ra**, **dtp**, etc.

 The important assumption is that the parent category provides information about which child labels are relevant.

 Instead of treating all labels as independent classes, the model explicitly represents the hierarchy.

---

 # Baseline: Flat Multi-Label Classification

 A simpler approach would be a flat architecture:

```
                 XLM-R
                   |
                   |
          shared representation h
                   |
                   |
          single classification head
                   |
                   |
        all label predictions
```

 In this case, XLM-R produces one representation of the input text. A single classifier predicts all labels:

 $$
y = sigmoid(W h + b)
$$

 The model has no explicit knowledge that some labels are parents and others are children.

 For example, it learns:

```
SP
it
IN
en
ra
dtp
...
```

 as a list of independent outputs.

 Even though the labels have a hierarchy, the classifier does not know that:

```
it belongs to SP
en belongs to IN
```

 unless it discovers this relationship implicitly from the training data.

 The advantages of this approach are simplicity and fewer parameters, but the limitation is that the model has no mechanism to specialize its representation based on the predicted parent category.

---

 # Proposed Architecture: Hierarchical Parent-Specific Mixture-of-Experts

 The idea is to introduce **parent-specific experts**.

 The architecture has three main components:

 1. A shared multilingual encoder (XLM-R)
2. A parent classifier that acts as a router
3. Parent-specific expert networks that create specialized representations for children

 The overall architecture is:

```
                  XLM-R
                    |
                    |
          shared representation h
                    |
        ┌───────────┴───────────┐
        |                       |
        |                       |
 Parent classifier        Parent-specific experts
        |                       |
        |                 ┌─────┼─────┐
        |                 |     |     |
        ▼                 ▼     ▼     ▼
 Parent probabilities    SP    NA    IN ...
        |
        |
        ▼
     Router weights
        |
        |
        ▼
 Weighted expert mixture
        |
        |
        ▼
 Shared child classifier
        |
        |
        ▼
 Child predictions
```

---

 # Step 1: Shared Representation

 The input text is first encoded by XLM-R:

 $$
h = XLM-R(x)
$$

 The representation $h$ captures general linguistic information from the text.

 This representation is shared by all later components.

---

 # Step 2: Parent Classification

 A parent classifier predicts the probability of each parent category:

 $$
p = sigmoid(W_p h + b_p)
$$

 For example:

```
SP = 0.85
NA = 0.10
IN = 0.70
OP = 0.05
```

 These probabilities are not only predictions. They are also used as **routing information**.

 The parent classifier therefore has two roles:

 1. Predict parent labels
2. Decide which experts should contribute to child prediction

---

 # Step 3: Parent-Specific Experts

 Each parent with children has its own expert network.

 For example:

```
SP expert
NA expert
HI expert
IN expert
OP expert
IP expert
```

 Each expert receives the same XLM-R representation:

 $$
E_p(h)
$$

 but transforms it differently.

 The idea is that different parent groups may require different features.

 For example:

 - The features useful for detecting internet-related categories may differ from features useful for interpersonal categories.
- The model can learn parent-specific transformations before predicting children.

---

 # Step 4: Mixture of Experts

 The parent probabilities determine how much each expert contributes.

 The final representation is a weighted combination:

 $$
h_{mix} =
\sum_p w_p E_p(h)
$$

 where:

 - $E_p(h)$ is the representation produced by expert $p$
- $w_p$ is the routing weight from the parent classifier

 Example:

 Suppose the parent classifier predicts:

```
IN = 0.8
SP = 0.7
NA = 0.1
```

 Then the final child representation may mostly combine:

```
IN expert contribution
+
SP expert contribution
```

 while ignoring weakly activated experts.

 The child classifier does not operate directly on the original XLM-R representation. Instead, it operates on this parent-aware mixture representation:

 $$
child\_logits = W_c h_{mix}+b_c
$$

---

 # Training Behavior

 During training, routing is soft.

 The model uses the parent probabilities directly:

 $$
w_p=p_p
$$

 This keeps the routing differentiable.

 Therefore, the child classification loss can update:

 - the child classifier
- the experts
- the parent classifier

 For example, if improving child prediction requires stronger IN expert activation, the gradient can flow back and adjust the parent router.

 This allows the parent classifier to learn not only parent prediction but also useful routing behavior.

---

 # Inference Behavior

 During evaluation but instead of predicting all labels with one flat classification head, I introduced parent-specific experts. A parent classifier predicts the relevant parent categories and acts as a router that determines how much each expert contributes. The expert representations are combined into a mixture representation, which is then used by a shared child classifier. Compared with a flat architecture, the model explicitly uses label hierarchy and allows different parent groups to, routing becomes harder.

 The parent probabilities are thresholded:

```
IN = 0.8  → active
SP = 0.7  → active
NA = 0.1  → inactive
```

 Only selected experts contribute to the final representation.

 This makes inference closer to a hierarchical decision process:

 1. Identify relevant parent categories
2. Activate corresponding experts
3. Predict children using the combined expert representation

---

 # Main Difference from a Flat Architecture

 The key difference is **conditional specialization**.

 ## Flat model

 The flat model learns:

```
Text representation
        |
        |
 Single classifier
        |
        |
All labels
```

 Every label prediction uses the same representation transformation.

 The model must learn all relationships inside one classifier.

---

 ## Hierarchical MoE model

 The proposed model learns:

```
Text representation
        |
        |
Parent prediction
        |
        |
Select relevant experts
        |
        |
Parent-aware representation
        |
        |
Child prediction
```

 The representation used for child prediction depends on the predicted parent structure.

---

 # Conceptual Advantage

 The assumption behind the model is:

 > Labels belonging to different parent categories may require different feature transformations, therefore the model should not force all child predictions through a single shared decision space.

 The experts provide specialized feature transformations, while the parent classifier determines which transformations are useful for each example.

---

 # Summary in One Paragraph

 "I built a hierarchical multi-label classifier where XLM-R provides a shared representation, but instead of predicting all labels with one flat classification head, I introduced parent-specific experts. A parent classifier predicts the relevant parent categories and acts as a router that determines how much each expert contributes. The expert representations are combined into a mixture representation, which is then used by a shared child classifier. Compared with a flat architecture, the model explicitly uses label hierarchy and allows different parent groups to learn specialized representations for their children."