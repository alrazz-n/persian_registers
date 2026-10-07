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

 This experiment implements a hierarchical multilabel classification approach based on XLM-R, in which the 25 target labels are divided into two related levels: 9 parent labels and 16 child labels. The main motivation for this architecture is to explicitly model the hierarchical structure of the classification scheme rather than treating all 25 labels as independent targets, as would be done in a conventional flat multilabel classifier.

The experiment uses the pretrained multilingual model TurkuNLP/web-register-classification-multilingual as the shared encoder. The input text is first tokenized with the XLM-R tokenizer and truncated or padded to a maximum sequence length of 512 tokens. For each input, the pretrained XLM-R encoder produces contextualized representations for all tokens. The representation of the first token, \<s>, is used as a shared document-level representation. This representation is then passed to two separate classification heads: one for the parent labels and one for the child labels. The parent classifier is a linear layer that maps the XLM-R hidden representation to 9 parent logits, while the child classifier is another linear layer that maps the same representation to 16 child logits. The two heads therefore share the entire XLM-R encoder but have separate output layers, allowing the model to learn information that is useful for both levels while also allowing parent- and child-specific decision boundaries.

The hierarchy is explicitly defined by associating each child label with its corresponding parent. For example, en, ra, dtp, fi, and lt belong to the parent IN, while rv, ob, rs, and av belong to OP. During preprocessing, the original dataset labels are first mapped to this hierarchical representation. Parent labels are also made consistent with their children: if any child belonging to a parent is positive, the corresponding parent is set to positive. Thus, the target representation contains 25 binary labels in a fixed order, with the 9 parent labels followed by the 16 child labels.

Unlike a flat multilabel model, where a single classification layer would directly predict all 25 labels and a single loss would normally be calculated over those predictions, this model produces two sets of predictions and calculates two separate binary cross-entropy losses. One loss measures the performance of the parent classifier, while the other measures the performance of the child classifier. These losses are then combined using independently tunable weights. The overall training objective is therefore a weighted combination of parent-level and child-level classification losses. This makes it possible to control how strongly the model is encouraged to learn the broader parent categories versus the more fine-grained child categories. In this experiment, both the parent and child loss weights are treated as hyperparameters rather than being fixed in advance.

The model is trained using the Hugging Face Trainer framework. The training, development, and test sets are kept separate, with the development set used for model selection and hyperparameter optimization and the test set reserved for the final evaluation. The evaluation metric used for model selection is child macro-F1, because the child labels represent the fine-grained classification task and macro-F1 gives each child category equal importance regardless of its frequency. This is particularly relevant when some of the fine-grained categories are less frequent than others.

Optuna is used to search for suitable training hyperparameters. The search includes the learning rate, weight decay, warmup ratio, gradient accumulation steps, and, importantly, the relative weights assigned to the parent and child losses. Each Optuna trial initializes a fresh XLM-R encoder and a new pair of classification heads, ensuring that the trials are independent. During training, the model is evaluated after each epoch. Early stopping is used to stop training when the development performance does not improve, while an Optuna pruning callback can terminate trials that perform poorly relative to the other trials. The best hyperparameter configuration is selected according to development-set child macro-F1.

After the Optuna search, a new model is initialized using the best hyperparameters and trained again. The best checkpoint according to development-set child macro-F1 is retained. The final model is then evaluated on the held-out test set. Predictions are obtained by applying the sigmoid function independently to each of the 25 logits, since this is a multilabel problem in which multiple labels can be active for the same document. A threshold of 0.5 is initially used to convert probabilities into binary predictions. Importantly, the experiment also saves the development and test logits, probabilities, labels, and predictions, making it possible to perform additional threshold optimization after training without retraining the model.

Performance is reported separately for the parent and child levels as well as for all 25 labels together. Macro-F1 and micro-F1 are calculated for the parent labels and child labels independently, in addition to overall macro-F1, micro-F1, and weighted-F1. A per-label classification report is also generated to examine how well the model performs on individual parent and child categories.

The main difference between this experiment and a flat baseline therefore lies in how the label structure is represented and how the training objective is constructed. A flat multilabel classifier would use the shared XLM-R representation followed by a single output layer containing all 25 labels, treating every label as an independent prediction target. In contrast, the hierarchical model separates the predictions into parent and child levels and uses two classification heads with separate loss terms. The parent labels provide an explicit representation of the broader categories, while the child labels capture the more specific distinctions within those categories. In addition, the weighted loss allows the training process to place different emphasis on these two levels. The encoder itself remains shared, so the model does not consist of two independent language models; instead, both classification tasks are learned jointly from the same multilingual representation.

Another important difference is that the hierarchical formulation incorporates structural information into the target representation. The preprocessing step enforces the logical relationship that a positive child implies a positive parent. Consequently, the parent labels are not treated as unrelated categories but are explicitly connected to their corresponding child categories. The model therefore has access to supervision at both the broad and fine-grained levels. This provides a way of investigating whether explicitly representing the label hierarchy improves fine-grained classification compared with a conventional flat 25-label multilabel model.

Overall, this experiment can be viewed as a shared-representation hierarchical multilabel classifier: XLM-R provides a common multilingual representation, two task-specific classification heads predict the parent and child categories, and a weighted combination of the two classification losses jointly trains the model. The experiment is designed to test whether exploiting the known parent–child structure of the label inventory can provide an advantage over treating all labels as independent outputs in a flat classification architecture.
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

 This experiment investigates a hierarchical, parent-conditioned approach to multilingual web-register classification using XLM-R, in contrast to the flat baseline model. In the flat baseline, all 25 labels are treated as independent targets and a single classifier predicts them directly from the shared XLM-R representation. In this experiment, the 25 labels are instead explicitly divided into two hierarchical levels: 9 parent categories and 16 child categories. The central idea is to first learn a representation that is useful for predicting the parent categories and then provide this parent-oriented representation to the child classifier. The model therefore retains the original XLM-R representation while additionally giving the child classifier access to information learned specifically for the parent-level task.

The label hierarchy is defined manually according to the structure of the dataset. The parent categories are MT, LY, SP, ID, NA, HI, IN, OP, and IP, while the child categories belong to specific parents; for example, it belongs to SP, ne, sr, and nb belong to NA, and en, ra, dtp, fi, and lt belong to IN. Some parent categories do not have children. This results in 9 parent labels and 16 child labels, giving 25 hierarchical outputs in total. An important aspect of the implementation is that the parent and child labels are kept in a fixed order throughout training, prediction, and evaluation. The final output vector therefore consists of the 9 parent logits followed by the 16 child logits.

The original dataset is loaded for each experimental dataset selected through the SLURM array index. The code reads the original label ordering from the dataset metadata and verifies that every label required by the hierarchical structure is present. The labels are then transformed into hierarchical targets. For the parent level, a parent is considered positive if either the original parent label is positive or at least one of its children is positive. This makes the parent labels consistent with the hierarchy: whenever a child belonging to a parent is active, its parent is also represented as active. The child labels themselves are retained from the original dataset. Unlike a flat model, therefore, the training target is not simply a list of independent labels; it explicitly represents two related levels of the label structure.

The input text is tokenized using the XLM-R tokenizer with a maximum sequence length of 512 tokens. The pretrained multilingual web-register model, TurkuNLP/web-register-classification-multilingual, provides the XLM-R encoder. For each input, the encoder produces contextualized token representations, and the representation of the first token is extracted as the document-level representation h. For XLM-R-large, this representation has 1024 dimensions. In the flat baseline, this shared representation can be passed directly to a classifier that predicts all labels. Here, however, the representation is processed in a hierarchical manner.

The first additional component is a parent projection layer. The 1024-dimensional XLM-R representation is passed through a linear layer that reduces it to 256 dimensions, followed by a GELU activation. This produces parent_hidden, a learned representation intended to capture information useful for distinguishing the parent categories. A parent classifier then maps this 256-dimensional representation to the 9 parent logits. Thus, the parent prediction pathway is h → parent projection → GELU → parent classifier, whereas the flat baseline would typically use a direct h → classifier pathway for all labels.

The main architectural difference from the flat model occurs when predicting the children. Instead of using only the original XLM-R representation, the child classifier receives a concatenation of the original 1024-dimensional representation h and the 256-dimensional parent_hidden representation. The resulting 1280-dimensional vector is passed to a child classifier that produces the 16 child logits. This means that the child classifier has access to both the general linguistic representation learned by XLM-R and an additional representation that has been shaped by the parent-level classification task. Importantly, the model does not feed the hard parent predictions into the child classifier. It uses the continuous learned parent_hidden representation instead. Consequently, the child pathway is differentiable and can be trained jointly with the parent pathway.

The two classification tasks are trained simultaneously using binary cross-entropy with logits. A separate loss is calculated for the 9 parent predictions and the 16 child predictions. The final training objective is a weighted sum of these two losses: the parent loss is multiplied by parent_weight and the child loss by child_weight. This differs from a flat model in which all labels would normally contribute directly to a single classification loss. Here, the two levels of the hierarchy have explicitly separate contributions to the optimization objective. The relative importance of learning the parent and child tasks is therefore controlled by the two loss weights.

The values of parent_weight and child_weight are not fixed manually. They are treated as hyperparameters and optimized with Optuna, together with the learning rate, weight decay, warm-up ratio, and gradient accumulation setting. For each Optuna trial, a new XLM-R encoder and a new hierarchical classification model are initialized, ensuring that the trials represent independent training runs rather than continuing from a previous trial. Each trial is trained for up to 10 epochs, with early stopping based on the validation performance. Optuna pruning is also used to stop trials that perform poorly during training. The optimization target is child-level macro-F1 on the development set, reflecting the experimental focus on obtaining balanced performance across the child categories rather than optimizing only the most frequent labels.

After the best hyperparameter configuration has been identified, a new hierarchical model is initialized using those parameters and trained again. The best model is selected according to development-set child macro-F1. The final model is then evaluated on the held-out test set. During evaluation, the parent and child logits are converted to independent probabilities using the sigmoid function. A threshold of 0.5 is initially applied to each probability to obtain binary multilabel predictions. Because this is a multilabel problem, sigmoid activation is appropriate: multiple labels can be positive for the same document, and the model does not have to choose exactly one class as would be the case with softmax classification.

The evaluation reports parent-level, child-level, and overall performance. Macro-F1 and micro-F1 are calculated separately for the parent labels and the child labels, as well as across all 25 labels. This makes it possible to determine whether the hierarchical architecture improves the parent task, the child task, or the overall classification problem. The per-label classification report and the raw logits, probabilities, labels, and predictions are also saved, allowing the predictions to be analyzed later without retraining the model.

The key difference between this experiment and the flat baseline is therefore not simply that the labels are grouped into parent and child categories. The model architecture and training objective are also changed to exploit this structure. In the flat model, the classifier receives the XLM-R representation and predicts all 25 labels without explicitly distinguishing between levels of the hierarchy. In this experiment, the model first develops a compact parent-oriented representation, uses it to predict the parent categories, and then combines that representation with the original XLM-R representation before predicting the children. At the same time, the parent and child objectives are optimized jointly, with their relative contributions controlled by separately optimized loss weights. The experiment consequently tests whether explicitly modeling the hierarchical relationship between the labels can provide useful information for child-level classification while preserving access to the full original XLM-R representation.

---

 # 5\. Experiment 3: Parent-Specific Expert Networks

Architecture:

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
        |                              │ NA     │ 256→3                 │ ne sr nb        │
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

 In this experiment, a hierarchical multilabel classification model was developed based on XLM-R, with the aim of explicitly modeling the relationship between parent and child labels. Unlike a flat baseline model, in which all labels are predicted independently from a single shared representation, this architecture separates the prediction process into a parent-classification branch and a set of parent-specific child-classification branches. The underlying pretrained encoder is TurkuNLP/web-register-classification-multilingual, which is based on XLM-R and produces a 1024-dimensional representation for each input text. For each example, the hidden representation corresponding to the first token (\<s>) is extracted from the final encoder layer and used as the shared representation for all subsequent predictions.

The label space consists of nine parent labels—MT, LY, SP, ID, NA, HI, IN, OP, and IP—and sixteen child labels distributed among six of these parents. Specifically, SP contains it, NA contains ne, sr, and nb, HI contains re, IN contains en, ra, dtp, fi, and lt, OP contains rv, ob, rs, and av, and IP contains ds and ed. MT, LY, and ID are parent-only categories and therefore do not have corresponding child labels. The resulting model produces 25 logits in total: nine parent logits and sixteen child logits.

Before training, the original dataset labels are transformed into this hierarchical representation. Parent labels are constructed so that a parent is considered positive either when it is explicitly annotated as positive in the original data or when at least one of its children is positive. This ensures consistency between the parent and child levels. The child labels retain their original annotations. The labels are then arranged in a fixed order, with the nine parent labels followed by the sixteen child labels, so that the output of the model and the target vectors always have the same structure.

The input texts are tokenized using the XLM-R tokenizer with a maximum sequence length of 512 tokens. Padding is dynamically handled by the data collator, with sequences padded to a multiple of eight to make the computation more efficient on the GPU. The dataset is divided into training, development, and test sets, and the development set is used during model selection and early stopping rather than for the final test evaluation.

The main architectural difference from a flat baseline is introduced after obtaining the shared 1024-dimensional XLM-R representation. The model first creates a separate parent representation by applying a linear projection from 1024 to 256 dimensions followed by a GELU activation. This 256-dimensional representation is passed to a single parent classifier that produces nine parent logits. Thus, parent prediction is performed through a dedicated branch rather than treating parent and child labels as a single undifferentiated set of 25 labels.

Child prediction is handled differently. Instead of using one common classifier for all sixteen child labels, the model contains a separate expert network for every parent that has children. Consequently, there are six parent-specific experts corresponding to SP, NA, HI, IN, OP, and IP. Each expert receives the same original 1024-dimensional XLM-R representation, rather than the parent prediction or the parent hidden representation. This is an important characteristic of the architecture: the experts do not perform hard routing based on the predicted parent. All relevant experts are executed for every example. This avoids the problem of making child prediction dependent on a potentially incorrect parent decision, which is particularly important because the task is multilabel and an example can contain multiple labels.

Each parent-specific expert consists of a 1024-to-256 linear transformation, GELU activation, dropout with a rate of 0.1, a second 256-to-256 linear transformation, and another GELU activation. The resulting 256-dimensional representation is therefore specialized for the corresponding parent category. A separate child classifier is then attached to each expert. For example, the NA expert produces three logits corresponding only to ne, sr, and nb, while the IN expert produces five logits corresponding to en, ra, dtp, fi, and lt. The child classifiers therefore operate on parent-specific representations rather than sharing one classifier across all child labels.

The outputs from the individual child experts are subsequently concatenated in a fixed global order to produce the sixteen child logits. These are concatenated with the nine parent logits to obtain the final 25-dimensional output. Although the model produces a single output vector in the end, the internal computation is explicitly hierarchical: the first nine outputs originate from the parent branch, while the remaining sixteen outputs originate from specialized child branches associated with their respective parents.

Training is performed as a multilabel classification problem using binary cross-entropy with logits. The loss is calculated separately for the parent and child predictions. A parent loss is computed by comparing the nine parent logits with the nine parent targets, while a separate child loss compares the sixteen child logits with the sixteen child targets. The two losses are then combined using independently tunable weights. In other words, the total objective is the weighted sum of the parent-level BCE loss and the child-level BCE loss. This allows the experiment to control the relative importance of learning the broader parent categories versus the more specific child categories.

This loss formulation is another difference from a conventional flat model. In a flat model, all labels would normally be predicted by a common classifier and optimized together as one flat label space. In the present experiment, parent and child supervision are explicitly separated, allowing the model to receive a dedicated training signal for the hierarchical structure. At the same time, the child classifiers remain multilabel classifiers rather than mutually exclusive classifiers, so multiple parent and child labels can be active for the same example.

The model is trained using the Hugging Face Trainer framework with mixed-precision BF16 training. Because the XLM-R encoder is large, gradient accumulation is used to obtain a larger effective batch size while maintaining a manageable per-device batch size. Early stopping is applied based on development-set child macro-F1, and the checkpoint achieving the best child macro-F1 is retained. Child macro-F1 is used as the primary selection criterion because the child labels are the more fine-grained part of the hierarchy and are therefore the main target of the parent-specific expert architecture.

Hyperparameters are selected using Optuna rather than being fixed manually. Eight trials are performed, with the search covering the learning rate, weight decay, warmup ratio, parent-loss weight, child-loss weight, and gradient accumulation steps. Each Optuna trial initializes a fresh model and trains it independently. The trial is evaluated after each epoch, and the child macro-F1 on the development set is reported to Optuna. A median-based pruning strategy can terminate trials that are performing poorly, reducing unnecessary computation. The best configuration is the one that obtains the highest development-set child macro-F1.

After hyperparameter optimization, a new model is initialized using the selected parent and child loss weights and the other best hyperparameters. This final model is then trained on the training data with the development set used for evaluation, early stopping, and model selection. Importantly, the test set is not used during hyperparameter optimization or model selection. After training is completed, the selected model is evaluated on the test set only once for the final performance assessment.

For evaluation, the model's logits are converted into probabilities using the sigmoid function, since the task is multilabel rather than multiclass. A default threshold of 0.5 is then applied independently to each of the 25 labels. Predictions and targets are evaluated separately at the parent and child levels. Parent macro-F1 and micro-F1 measure performance on the nine parent categories, while child macro-F1 and micro-F1 measure performance on the sixteen child categories. Overall macro-, micro-, and weighted-F1 scores are also calculated across the complete 25-label output space. The individual predictions, probabilities, logits, targets, classification report, label hierarchy, and selected hyperparameters are saved so that the experiment can be analyzed or reproduced later.

The key conceptual difference between this experiment and a flat base model is therefore not simply the number of output labels, but how the model learns representations for those labels. A flat model would take the shared XLM-R representation and pass it through one common classification layer that independently predicts all 25 labels. In contrast, this experiment explicitly separates the parent and child tasks and gives each group of child labels its own expert network. The six experts can therefore learn different transformations of the shared XLM-R representation that are specialized to the semantic characteristics of their corresponding parent categories. For example, the representation used to distinguish the five IN children does not have to be identical to the representation used to distinguish the four OP children. At the same time, because every expert receives the same shared XLM-R representation and all experts are evaluated for every example, the architecture does not rely on a potentially error-prone hard routing decision from the parent classifier. Overall, the experiment can therefore be viewed as a hierarchical multilabel architecture with shared multilingual language understanding at the encoder level, a dedicated parent prediction branch, and multiple parent-specific expert networks for fine-grained child prediction, providing a more structured alternative to treating all 25 labels as a single flat label space.

---

 # 6\. Experiment 4 and 5: Hard-Routed Hierarchical Experts


---

 # 6.1\. Experiment 4: Gold-Routed Trainind - Hard-Routed Evaluation Hierarchical Experts


```
                          XLM-R encoder (shared)
                                  │
                                  v
                       h = CLS embedding [1024]
                                  │
              ┌───────────────────┴────────────────────┐
              │                                        │
              v                                        │
      ┌───────────────────┐                            │
      │    PARENT HEAD    │                            │
      │ Linear 1024→256   │                            │
      │      + GELU       │                            │
      └─────────┬─────────┘                            │
                │                                      │
                v                                      │
         Linear 256→9                                  │
         9 parent logits                               │
      (MT LY SP ID NA HI IN OP IP)                     │
                │                                      │
       ┌────────┴──────────────────┐                   │
       │                           │                   │
   TRAINING                    EVAL / TEST             │
   (labels given)             (dev + final test)       │
       │                           │                   │
       v                           v                   │
  GOLD PARENTS               PREDICTED PARENTS         │
  parent annotated          sigmoid(logit) ≥ 0.5       │
  positive OR                                          │
  any child annotated                                  │
  positive                                             │
       │                           │                   │
       └──────────────┬────────────┘                   │
                      v                                │
          ACTIVE PARENT MASK [batch, 9]                │
          (multi-label: several can be 1)              │
                      │                                │
                      │  Example:                      │
                      │  Gold parents = {NA, IN}       │
                      │  mask = [0,0,0,0,1,0,1,0,0]    │
                      │                 ↑     ↑        │
                      │                NA    IN        │
                      │                                │
                      │  → NA + IN experts activated   │
                      │  → same example → both experts │
                      │                                │
                      │  Evaluation example:           │
                      │  P(NA)=0.91 → active           │
                      │  P(IN)=0.87 → active           │
                      │  P(OP)=0.12 → inactive         │
                      │                                │
                      └──────────────┬─────────────────┘
                                     v
              For each parent with children:
              h_sel = h[rows where mask = 1]
              (only rows selected; h is unchanged)
                                     │
       ┌────────┬────────┬───────────┼────────┬────────┬────────┐
       v        v        v           v        v        v
      SP       NA       HI          IN       OP       IP
       │        │        │           │        │        │
       v        v        v           v        v        v
                    Parent-specific Expert MLP
             separate weights for each parent
       │        │        │           │        │        │
       │        │        │           │        │        │
       └────────┴────────┴───────────┴────────┴────────┘
                                     │
                    Linear 1024→256 → GELU
                                     │
                                  Dropout
                                     │
                    Linear 256→256 → GELU
                                     │
       ┌────────┬────────┬───────────┼────────┬────────┬────────┐
       v        v        v           v        v        v
    256→1    256→3    256→1       256→5    256→4    256→2
       │        │        │           │        │        │
      it     ne sr nb     re       en ra    rv ob     ds ed
                                  dtp fi     rs av
                                  lt
```

```
             MT, LY, ID: parent-only
                   no expert / no children
                                     │
                                     v
                  SCATTER into full child tensor
                                     │
                 child_logits = [batch, 16], -20
                                     │
              routed positions ← expert outputs
              inactive positions remain at -20
                        sigmoid(-20) ≈ 0
                                     │
                                     v
             [ 9 parent logits | 16 child logits ]
                                     │
                                     v
                              25 output logits

```

```
LOSS (only when labels are given)

  parent_loss = BCE(9 parent logits, parent labels)       all examples
  child_loss  = mean over active experts of
                BCE(that expert's child logits,
                    that expert's child labels)           routed rows only
  total       = w_parent · parent_loss + w_child · child_loss
```

Example for h_sel:

```

h = [example 1
     example 2
     example 3
     example 4]

NA mask = [1, 0, 1, 0]

h_sel(NA) = [example 1
             example 3]

```
  Routing in the loss: gold in train mode, predicted in eval mode.


This experiment, referred to as GoldRout, extends the flat multi-label classification approach by explicitly incorporating the hierarchical structure of the labels. The underlying text representation is obtained from the same multilingual XLM-R encoder, but instead of predicting all 25 labels directly from a single classification layer, the model first predicts the nine parent categories and then uses these parent categories to determine which specialized child classifiers should process each example. The main purpose of this experiment is therefore to investigate whether exploiting the known label hierarchy can improve child-level classification compared with a flat model that treats all labels as independent outputs.

The model starts with the pretrained TurkuNLP/web-register-classification-multilingual encoder. For each input document, the XLM-R encoder produces contextualized token representations, and the representation of the first token (the \<s>/CLS-style representation) is used as the shared document-level representation. This produces a 1024-dimensional vector for each example. In contrast to a flat model, where this representation would typically be passed directly to one classifier producing all 25 label logits, GoldRout uses the representation in two related but separate stages.

First, the shared 1024-dimensional representation is passed through a 1024→256 linear projection followed by a GELU activation. This produces a compact parent-level representation. A second linear layer maps this representation to nine parent logits, corresponding to MT, LY, SP, ID, NA, HI, IN, OP, and IP. These parent predictions are treated as independent binary decisions because the task is multi-label: an example may belong to several parent categories simultaneously. Importantly, the parent labels are not mutually exclusive. For example, an example can simultaneously belong to NA and IN, meaning that both parent categories can be active for the same document.

The parent labels are also constructed hierarchically from the original annotations. A parent is considered positive when it is explicitly annotated as positive or when at least one of its children is positive. This ensures that the parent-level representation is consistent with the child annotations. For example, if an example has the child label ne, its parent NA is automatically considered positive even if NA itself was not explicitly annotated. Parent-only categories such as MT, LY, and ID do not have child labels and therefore do not require a separate child expert.

The main architectural difference from the flat baseline begins after the parent predictions have been produced. GoldRout uses hard parent routing. During training, the routing decision is based on the gold parent labels rather than the model's predicted parent probabilities. Thus, if an example has NA and IN as positive parents, that example is routed simultaneously to both the NA expert and the IN expert. This is important because the routing is multi-label rather than exclusive: an example is not forced to select only one parent. Each positive parent activates its corresponding expert independently.

Only parents that have children have dedicated experts. In this experiment, SP, NA, HI, IN, OP, and IP therefore receive their own expert networks. MT, LY, and ID are parent-only categories and are predicted directly by the parent classifier without an additional child classifier. Each expert receives the original 1024-dimensional XLM-R representation rather than the parent projection. The experts are separate neural networks with their own parameters. Each consists of a 1024→256 linear layer, GELU activation, dropout, followed by another 256→256 linear layer and GELU activation. Consequently, the model does not use the same child-processing network for all parents. Instead, each parent has a specialized representation that can learn features particularly relevant to distinguishing its own children.

For example, the NA expert is responsible only for distinguishing ne, sr, and nb, whereas the IN expert is responsible for en, ra, dtp, fi, and lt. These experts are therefore exposed to different subsets of the training examples and have different parameters. This is a substantial difference from a flat classifier, where all 16 child labels would be predicted by the same output layer and would share the same final classification parameters.

After an example has been routed to a parent expert, the expert output is passed to a parent-specific linear classifier whose number of outputs corresponds exactly to the number of children of that parent. Thus, the NA expert produces three child logits, the HI expert produces one, the IN expert produces five, the OP expert produces four, the IP expert produces two, and the SP expert produces one. The outputs from these different experts are then placed back into a common 16-dimensional child-logit tensor so that the final model still produces a consistent 25-label output: nine parent logits followed by sixteen child logits.

For examples that are not routed to a particular parent, the corresponding child logits are initialized to a very negative value (-20). After applying the sigmoid function, this produces a probability very close to zero. This means that an inactive parent effectively suppresses all of its child predictions. For example, if an example is not routed to OP, the four OP children (rv, ob, rs, and av) receive approximately zero probability. This is another important difference from a flat model: in the flat model, every child classifier produces a prediction for every example, whereas in GoldRout, child predictions are conditional on the corresponding parent being active.

The training objective also reflects this hierarchical structure. The parent classification loss is calculated using binary cross-entropy over all nine parent labels for every training example. This allows the model to learn the parent-level routing decisions. The child loss, however, is calculated only for the experts that are active for a given example. In training, the gold parent labels determine this active set. For instance, if an example has NA and IN as positive parents, its child labels are used to calculate the loss for both the NA and IN experts, while the HI, OP, IP, and SP experts do not contribute a child loss for that example. The individual parent-specific child losses are averaged and then combined with the parent loss using separately tunable parent and child loss weights. The total objective can therefore be expressed conceptually as:

Total loss = parent weight × parent loss + child weight × child loss.

This loss design allows the experiment to investigate whether explicitly supervising the parent decisions while simultaneously training specialized child classifiers provides an advantage over treating all labels as a single flat prediction problem.

The routing mechanism is deliberately different between training and evaluation. During training, gold parent labels are used for routing. This provides the child experts with the correct parent context and prevents incorrect early parent predictions from blocking the corresponding child expert during learning. In other words, if an example is genuinely an NA example, the NA expert is trained on it even if the parent classifier has not yet learned to predict NA correctly. This makes the training procedure effectively a form of gold routing. During development and test evaluation, however, gold parent labels are not available for routing. The model instead applies the sigmoid function to the predicted parent logits and activates a parent when its predicted probability reaches the specified routing threshold of 0.5. Consequently, evaluation reflects the complete prediction pipeline: the model must first identify the relevant parents and then classify the children under those predicted parents.

This distinction is important when interpreting the experiment. The model is therefore not simply a collection of independent child classifiers trained with known parent information. At test time, child predictions depend on the model's own parent predictions. An incorrect parent decision can consequently affect the downstream child predictions. For example, if the model fails to activate IN for an example whose correct child is en, the IN expert is not executed for that example and en cannot be predicted as positive. Conversely, activating an irrelevant parent causes its expert to run and potentially produce child predictions even though the parent itself is incorrect. The experiment therefore evaluates whether the benefits of hierarchical specialization outweigh the additional dependency introduced by routing.

The parent routing threshold and child prediction threshold are kept conceptually separate. The parent routing threshold determines which experts are executed, whereas the child threshold determines whether an individual child probability is converted into a positive prediction. In the current experiment both are initially set to 0.5, but keeping them as separate parameters allows them to be tuned independently in later analysis.

The optimization procedure also differs from simply selecting a fixed set of hyperparameters. Optuna is used to search over the learning rate, weight decay, warm-up ratio, gradient accumulation, parent loss weight, and child loss weight. Each Optuna trial creates a fresh XLM-R-based hierarchical model and trains it using the training split while evaluating on the development split. The main optimization criterion is child macro-F1, reflecting the focus of the experiment on performance across the individual child categories rather than allowing frequent child labels to dominate the evaluation. Early stopping is also used to terminate training when the development performance no longer improves, while Optuna pruning can terminate trials that are performing poorly relative to other trials.

After the best hyperparameter configuration is identified, a new model is initialized and trained using those parameters. The resulting model is then evaluated on the development and test sets. Predictions and logits are saved so that the decision thresholds can subsequently be analyzed or tuned without having to retrain the model. The final evaluation reports parent-level, child-level, and overall macro- and micro-F1 scores, together with a per-label classification report.

Conceptually, therefore, the main difference between GoldRout and the flat baseline is where and how the label dependencies are represented. A flat model uses the shared XLM-R representation to make all 25 predictions directly, treating the parent and child labels as outputs of the same general classification problem. GoldRout instead decomposes the task into a parent prediction problem followed by parent-specific child prediction problems. The parent classifier determines which semantic regions of the label hierarchy are relevant, and separate experts specialize in distinguishing the children within those regions. Because multiple parents can be active simultaneously, the architecture retains the multi-label nature of the original task while still exploiting the hierarchical organization of the labels.

The experiment can therefore be viewed as testing the hypothesis that hierarchical specialization improves fine-grained classification: rather than asking one classifier to distinguish all child categories simultaneously, the model first identifies the relevant parent categories and then allows separate expert networks to focus on the much smaller and semantically related set of children belonging to each parent. At the same time, the use of gold parent routing during training isolates the effect of expert specialization by ensuring that the child experts receive the appropriate training examples, while predicted routing during evaluation tests how well the complete hierarchical system operates under realistic inference conditions.  
---

# 6.2\. Experiment 5: Hard-Routed Trainind - Hard-Routed Evaluation Hierarchical Experts

```

XLM-R encoder
    │
    ▼
h = first-token (<s>) representation [1024]
    │
    ├─────────────────────────────────┐
    │                                 ▼
    │                            Parent head
    │              Linear(1024→256) → GELU → Linear(256→9)
    │                                 │
    │                                 ▼
    │                         Parent logits [9] ───────────┬────────────────┐
    │                                 │                    │                │
    │                                 ▼                    ▼                │
    │                  Sigmoid → Threshold ≥ 0.5      Parent BCE            │
    │                                 │            (vs gold parents [9])    │
    │                                 ▼                                     │
    │                        Active parent mask                             │
    │                 (no gradient through threshold)                       │
    │                                 │                                     │
    │      For each parent with children (SP, NA, HI, IN, OP, IP):          │
    │                        ┌────────┴────────┐                            │
    │                   mask = 1           mask = 0                         │
    │                        │                 │                            │
    │                        ▼                 │                            │
    └──────────────────► Expert MLP(h)         │                            │
                             │                 │                            │
                             ▼                 │                            │
                        Child head             │                            │
                             │                 │                            │
                             ▼                 ▼                            │
                      Routed child       child logits = -20                 │
                         logits          (inactive slots)                   │
                             │                 │                            │
                             └────────┬────────┘                            │
                                      ▼                                     │
                             Child logits [16] ──────► Child BCE            │
                                      │         (per expert, on its own     │
                                      │          children and routed        │
                                      │          examples; mean over        │
                                      │          active experts)            │
                                      ▼                                     │
                          ┌─────────────────────┐                           │
                          │     CONCATENATE     │◄──────────────────────────┘
                          │[parent 9]+[child 16]│
                          └──────────┬──────────┘
                                     │
                                     ▼
                             Output logits [25]
                         (used for predictions/metrics)

```

```
Total loss:
    L = w_parent · Parent BCE + w_child · Child BCE
```

This experiment implements a hierarchical multi-label classification model based on XLM-R, in which the prediction process is explicitly divided into parent-level and child-level decisions. Unlike a conventional flat multi-label baseline, where all labels are predicted independently from the same encoder representation using a single classification layer, this approach introduces the label hierarchy into both the model architecture and the training procedure. The purpose is to investigate whether modeling the relationships between parent and child labels can improve the prediction of the more fine-grained child categories.

The model uses a pretrained multilingual XLM-R encoder as its shared representation layer. For each input text, the representation of the first token (\<s>) is extracted and used as the document-level representation. This representation is then passed to a parent classification head consisting of a linear projection from the XLM-R hidden size to 256 dimensions, a GELU activation, and a second linear layer that produces nine parent logits. Since the task is multi-label, the nine parent categories are treated as independent binary decisions rather than as mutually exclusive classes. A sigmoid function converts the parent logits into probabilities, and a threshold of 0.5 determines which parents are considered active.

The parent predictions play an important role in the remainder of the architecture. Six of the nine parent categories have associated child labels: SP, NA, HI, IN, OP, and IP. Each of these parents is assigned its own expert network and its own child classification head. The remaining parents (MT, LY, and ID) do not have children and therefore do not require an expert. For an input example, only the experts corresponding to the parents predicted as active are executed. Consequently, an example can be routed to multiple experts simultaneously because the task is multi-label. For instance, if the model predicts both NA and IN as active, the representation of the input is passed to both the NA expert and the IN expert, and each expert independently predicts the children belonging to its parent.

Each parent-specific expert receives the original XLM-R representation rather than the output of the parent classification layer. The expert consists of two linear transformations with a GELU activation and dropout, producing a 256-dimensional expert representation. This representation is then passed to the corresponding child classifier. Because each expert is responsible only for the children belonging to one parent, its output dimensionality is different according to the number of children associated with that parent. The NA expert, for example, predicts ne, sr, and nb, whereas the IN expert predicts en, ra, dtp, fi, and lt. This creates a set of specialized child prediction pathways instead of a single classifier responsible for all child labels.

Although the child predictions are produced by separate experts, they are reconstructed into a common 16-dimensional child-logit vector so that the final model output maintains a fixed and consistent label ordering. The complete output therefore contains nine parent logits followed by sixteen child logits, giving a total of 25 output dimensions. When a parent is not activated for a particular example, the corresponding child logits are set to a very negative value (-20). After applying the sigmoid function, these values correspond to probabilities very close to zero. Thus, an inactive parent effectively prevents its children from being predicted. This is the main hard-routing mechanism of the experiment: the parent prediction determines which child classifiers are allowed to produce predictions.

Importantly, routing is based on the model's predicted parent probabilities rather than on the gold parent labels. During both training and evaluation, the sigmoid probabilities of the parent classifier are compared with the parent routing threshold of 0.5. This means that the model does not receive the correct parent category as an input to the child prediction stage. Instead, it must first learn to predict the parent and then use its own prediction to determine which child expert is activated. This makes the training and inference procedures consistent and avoids using gold-label information during training that would not be available at test time. Because the threshold operation is non-differentiable, however, the child loss cannot propagate gradients through the routing decision back into the parent classifier. The parent classifier is therefore trained directly through its own parent-level loss, while the child experts are trained through the child-level loss for the examples routed to them.

The target labels are constructed to explicitly reflect the hierarchy. A parent is considered positive if it is explicitly positive in the original annotation or if at least one of its children is positive. This ensures that the parent labels are consistent with the child labels. The child labels themselves retain their original binary annotations. The resulting target vector contains the nine parent labels followed by the sixteen child labels, matching the model's output structure.

Training uses two binary cross-entropy losses with logits. The first is the parent loss, which is calculated over all nine parent labels for every training example. The second is the child loss, which is calculated only for the child experts that were activated by the predicted parent routing decisions. For each active parent, the model compares the expert's child logits with the corresponding child labels. The losses from the active experts are averaged to obtain the overall child loss. The final training objective is a weighted combination of the two components,

L=wparentLparent+wchildLchild,

where the parent and child weights are treated as hyperparameters. This allows the experiment to investigate how much emphasis should be placed on learning the coarse parent categories versus the more fine-grained child categories.

The hyperparameters are optimized using Optuna rather than being fixed in advance. The search considers the learning rate, weight decay, warmup ratio, gradient accumulation, and, importantly, the relative weights of the parent and child losses. Each Optuna trial initializes a fresh XLM-R encoder and a new hierarchical model, trains it on the training set, and evaluates it on the development set. The child macro-F1 score is used as the optimization objective because the main interest of this experiment is the performance of the fine-grained child categories, including categories that may be less frequent than others. Early stopping and Optuna pruning are used to avoid spending computational resources on trials that are unlikely to produce competitive results. After the best hyperparameter configuration is identified, a new model is initialized and trained using those parameters.

Evaluation is performed separately for the parent and child levels as well as for the complete 25-label output. Parent macro-F1 and micro-F1 measure how well the model identifies the broader categories, while child macro-F1 and micro-F1 measure the performance of the specialized child predictions. Overall macro-, micro-, and weighted-F1 scores are also calculated across all labels. In addition, the experiment records routing statistics, showing how frequently each parent expert is activated during evaluation. These statistics are useful for understanding how the hard-routing mechanism distributes examples across the different experts.

The main difference between this model and a flat multi-label baseline is therefore not simply the number of layers, but the way the label structure is incorporated into the prediction process. In a flat baseline, the XLM-R representation is typically passed directly to a single classification layer that produces all 25 label logits. Every label is predicted from the same shared representation, and the model does not explicitly distinguish between parent and child decisions. A flat model also does not use a parent prediction to determine whether a particular child classifier should be active. Consequently, all child labels are treated as independent outputs even though some of them belong to specific parent categories.

In contrast, the proposed hard-routed model first predicts the parent categories and then uses these predictions to selectively activate parent-specific child experts. The model therefore introduces an explicit conditional structure: child prediction depends on the predicted parent activation. This can potentially make the child prediction problem easier because each expert only needs to discriminate between the children associated with its own parent rather than learning all child categories simultaneously. It also gives the model a form of specialization, since different experts can learn different linguistic patterns associated with different parts of the label hierarchy. At the same time, the hard-routing mechanism introduces an important trade-off that does not exist in the flat model: an incorrect parent prediction can prevent the corresponding child expert from being activated, making it impossible for that child to be predicted correctly. Thus, the hierarchical model can benefit from specialization and structural information, but its child-level performance is also dependent on the quality of the parent-level routing.

Overall, this experiment can be viewed as a hierarchical, conditionally activated extension of the flat XLM-R multi-label classifier. The encoder remains shared across the entire task, while the prediction process is divided into a general parent classifier and a set of specialized parent-specific child experts. The experiment therefore tests whether explicitly exploiting the known parent-child label structure, together with hard routing based on predicted parent probabilities, provides an advantage over treating all labels as a single flat multi-label prediction problem.
---
 # 7\. Experiment 6: Soft-Routed Trainind - Hard-Routed Evaluation Hierarchical Experts



```
GOLD LABELS: 25 = 9 parents + 16 children
(parent = 1 if explicitly annotated OR any child = 1)

B = batch size

                 XLM-R ENCODER (shared)
                           │
                           ▼
            h = last_hidden_state[:, 0]
                     [B, hidden]
                           │
             ┌─────────────┴─────────────────────────────┐
             ▼                                           ▼
      PARENT BRANCH                              6 PARENT EXPERTS
      (all 9 parents)                            (take h)
                                                 SP(1) NA(3) HI(1)
      Linear(hidden → 256)                       IN(5) OP(4) IP(2)
             │                                   MT, LY, ID: no expert
            GELU                                 (parent logits only)
             │                                          │
      Linear(256 → 9)                        each expert:
             │                               Linear(hidden → 256)
             ▼                                          │
      parent_logits [B,9]                              GELU
             │                                          │
             ├────────► Parent BCE loss            Dropout(0.1)
             │                                          │
             ▼                                   Linear(256 → 256)
          sigmoid                                       │
             │                                         GELU
             ▼                                          │
    p(parent) [B,9]                                     ▼
             │                                   child classifier
             │                                   Linear(256 → n_children)
             │                                          │
             └──────────────────┬───────────────────────┘
                                ▼
                 child_logits initialized to -5.0
                           [B,16]
                                │
                  ┌─────────────┴──────────────┐
                  ▼                            ▼
        TRAINING: SOFT ROUTING        DEV / TEST: HARD ROUTING
                  │                            │
        Every expert runs on          For each parent:
        every example.                p(parent) >= 0.5 ?
                  │                            │
                  │                      ┌─────┴─────┐
                  │                     YES           NO
                  │                      │            │
                  ▼                      ▼            ▼
        expert_logits × p(parent)   Expert runs   Block remains
                  │                 on selected    at -5.0
        p is NOT detached           examples only  (~0.007 prob.)
        → child loss flows               │
          into parent head               ▼
                  │                Write raw expert logits
        NOTE: logit × p → 0        into selected positions
        means probability 0.5,     (NO × p)
        not 0                            │
                  │                Several parents can be
                  ▼                active at the same time
        Write weighted logits            │
        into child_logits                │
                  │                      │
                  └───────────┬──────────┘
                              ▼
                    child_logits [B,16]
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
          Child BCE loss          concat [parent 9 | child 16]
                 │                 = all_logits [B,25]
                 ▼                         │
          TOTAL LOSS                       ▼
          = parent_weight ×              sigmoid
            parent BCE                     │
          + child_weight ×         ┌───────┴────────┐
            child BCE              ▼                ▼
                                parent prediction  child prediction
          (computed whenever         │                │
           labels are given;     threshold 0.5    threshold 0.5
           optimised only in         │                │
           training)                 └───────┬────────┘
                                             ▼
                                       F1 metrics /
                                       saved logits

          NOTE: per-epoch validation (early stopping, best model on
          eval_child_macro_f1) also uses the HARD routing path,
          although the weights are trained with SOFT routing.
```


IMPORTANT TRAINING GRADIENT PATH:

```
        Child BCE
           ↓
        child logits
           ↓
        expert logits × p(parent)
                         ↓
                  parent classifier

        Therefore child loss also trains
        the parent classifier.

```

EVALUATION:
```


        parent probabilities → hard routing
        → inactive child logits remain -5
        → sigmoid(-5) ≈ 0.0067
```
This experiment implements a hierarchical multi-label classification model based on XLM-R, in which the prediction process is explicitly structured around a hierarchy of parent and child labels. Unlike a conventional flat multi-label classifier, where all labels are predicted independently from a single shared representation, this model first predicts a set of parent-level categories and then uses those predictions to determine how the child-level classification is performed. The purpose of this experiment is therefore not only to predict the final labels, but also to incorporate the hierarchical relationships between labels into the architecture and learning process.

The model uses the pretrained multilingual XLM-R encoder, which is shared across all labels and produces a contextual representation of the input text. The representation corresponding to the first token is used as the document-level representation. From this shared representation, the model creates a separate parent representation through a small feed-forward network consisting of a linear projection from the XLM-R hidden dimension to 256 dimensions followed by a GELU activation. This representation is passed to a parent classifier that produces nine independent parent logits, corresponding to MT, LY, SP, ID, NA, HI, IN, OP, and IP. Since this is a multi-label problem, the parent predictions are not mutually exclusive: several parents can be predicted as active for the same text. A sigmoid function converts the parent logits into independent parent probabilities.

The main difference from a flat baseline begins at this point. In a flat model, the shared XLM-R representation would normally be passed directly to a single classifier producing all 25 label logits, with every label being treated at the same level. The hierarchical model instead separates the parent prediction task from the child prediction task and introduces parent-specific experts for the child labels. Only parents that have child categories receive an expert network. Thus, parents such as MT, LY, and ID, which do not have children in the defined hierarchy, are predicted only at the parent level, whereas SP, NA, HI, IN, OP, and IP each have their own expert responsible for predicting their associated child labels.

Each parent-specific expert receives the original XLM-R representation rather than the parent representation used by the parent classifier. The expert consists of a small feed-forward network with a 256-dimensional hidden layer, GELU activation, dropout, a second linear transformation, and another GELU activation. The resulting expert representation is then passed to a parent-specific child classifier. Consequently, the model does not have one common classifier for all child labels. Instead, each group of related children is handled by a specialized classifier. For example, the IN expert predicts the five IN children, while the OP expert predicts the four OP children. This provides the model with a mechanism for learning different representations for different regions of the label hierarchy.

The routing mechanism is the key component of this experiment. During training, the model uses soft, differentiable routing rather than routing examples exclusively according to their predicted parent labels. Every parent-specific expert processes every training example. However, the output of each expert is multiplied by the predicted probability of its corresponding parent. If the model assigns a high probability to a parent, the corresponding expert has a stronger contribution to the child logits. If the parent probability is low, the expert's contribution is correspondingly reduced. This makes the routing mechanism differentiable and allows the child classification objective to influence the parent classifier. In other words, the parent classifier is not trained only through the parent classification loss: the child loss can also propagate through the parent probabilities because those probabilities determine the strength of the child experts' contributions.

An important consequence of this design is that the multiplication is performed on the child logits rather than on the child probabilities. Therefore, a parent probability of 0.5 does not make the child probability 0.5. Instead, it scales the child logit toward zero. For example, if an expert produces a positive child logit and the parent probability is 0.5, the resulting child logit is half as large, after which the sigmoid function is applied. This distinction is important when interpreting the routing mechanism: the parent probability acts as a continuous gating factor on the expert's evidence rather than directly representing the child's probability.

The child logits are initially set to -5 for all child labels. This provides a default inactive state for child labels that are not produced by an active routing path. A value of -5 corresponds to a sigmoid probability of approximately 0.0067, so these labels have a very low predicted probability rather than an exactly zero probability. During soft training, the logits corresponding to each parent are replaced by the weighted outputs of that parent's expert. The complete child-logit vector is then concatenated with the nine parent logits, producing the final 25-label output vector.

The training objective contains two binary cross-entropy losses with logits: one for the parent labels and one for the child labels. The parent loss measures how well the model predicts the nine parent categories, while the child loss measures the performance of the routed child classifiers. The two losses are combined using independently tunable parent and child weights. These weights are optimized as part of the hyperparameter search. This differs from a flat model because the child classification objective is connected structurally to the parent classification process through the soft routing probabilities. Therefore, errors at the child level can affect the learning of the parent representation and parent classifier.

The parent targets are also constructed hierarchically. A parent is considered positive if it is explicitly annotated as positive in the original data or if at least one of its children is positive. This ensures consistency between the parent and child levels: a positive child automatically implies a positive parent. The child labels themselves retain their original annotations. The resulting target vector contains the nine parent labels followed by the sixteen child labels, giving a total of 25 outputs.

At evaluation time, the routing mechanism changes from soft routing to hard routing. The model first predicts the parent probabilities and compares each probability with a threshold of 0.5. Each parent whose predicted probability reaches this threshold is considered active. The corresponding parent-specific expert is then applied only to the examples for which that parent is active. Several parents can be active for the same example because the task is multi-label. For example, an example could simultaneously activate the IN and NA experts. Child logits belonging to inactive parents remain at the default value of -5, while the logits produced by active experts are inserted into their corresponding positions. This creates a sparse prediction mechanism in which the parent predictions determine which child classifiers are allowed to contribute to the final output.

This training/evaluation distinction is important. The model is trained with soft routing so that the entire routing mechanism remains differentiable and the child loss can influence the parent classifier. During development and testing, however, the model uses hard routing to evaluate the actual hierarchical decision process. Therefore, validation and test performance reflect the consequences of making discrete parent routing decisions rather than using the softer training approximation.

The model is trained using the Hugging Face Trainer framework with XLM-R fine-tuning. Learning rate, weight decay, warmup ratio, gradient accumulation, parent loss weight, and child loss weight are treated as hyperparameters and optimized using Optuna. The optimization objective is child macro-F1 on the development set, which places particular emphasis on performance across the individual child categories rather than allowing frequent labels to dominate the optimization criterion. Early stopping is also used during training, with the best model selected according to development child macro-F1.

The final model is trained using the best hyperparameters discovered by Optuna and is subsequently evaluated on the held-out test set. In addition to overall macro- and micro-F1, the experiment separately reports parent-level and child-level performance. Parent routing statistics are also recorded to show how frequently each parent expert is activated. This is useful for understanding whether some experts are rarely or excessively selected and therefore provides information about the behavior of the learned routing mechanism.

The fundamental difference from a flat baseline is therefore architectural and not simply a change in the number of layers. In a flat model, the XLM-R representation is normally passed to one classifier that predicts all labels independently, meaning that the model has no explicit mechanism requiring it to first identify a parent category before making a child prediction. In this hierarchical experiment, the model first learns parent-level probabilities, maintains separate expert networks for different parent categories, and uses the parent predictions to control the contribution of those experts to child predictions. The hierarchy is consequently incorporated directly into both the model architecture and the optimization process.

Another important distinction is that the hierarchical model does not simply use the predicted parent as a preliminary classification step and then make a child prediction afterward. Because the routing is multi-label, multiple parent experts can contribute to the same example. During training, all relevant experts are evaluated and their outputs are continuously weighted by the corresponding parent probabilities. This allows the model to represent uncertainty about the parent category and to gradually transition between experts rather than making a completely discrete decision during optimization. At inference time, this soft mechanism is converted into hard routing using the 0.5 parent threshold.

Overall, this experiment can be viewed as a hierarchical, multi-label mixture-of-experts approach built on top of XLM-R. The shared encoder provides a common multilingual representation, the parent branch learns the high-level structure of the label space, and parent-specific experts specialize in predicting the children associated with each parent. Soft routing during training provides a differentiable connection between parent and child classification, while hard routing during validation and testing evaluates the model under the intended hierarchical decision process. In contrast to the flat baseline, where all labels compete within a single undifferentiated output layer, this model explicitly exploits the known relationships between labels and allows different groups of child labels to be modeled by specialized expert networks.

---

 # 8\. Experiment 7: Parent-Specific Mixture-of-Experts


 The architecture:


```

    XLM-R encoder
          │
    h = last_hidden_state[:, 0]   (1024)
          │
     ┌────┴───────────────────────────────┐
     │                                     │
     ▼                                     ▼
 PARENT HEAD                         6 PARENT-SPECIFIC
 Linear 1024→256                     EXPERTS
      │                              SP, NA, HI, IN, OP, IP
    GELU                                  │
      │                                   │
 Linear 256→9                             │
      │                              each expert:
      ▼                              Linear 1024→256
 parent logits (9)                        │
      │                                  GELU
 sigmoid                                  │
      │                               Dropout(0.1)
      ▼                                   │
 parent probabilities p (9)           Linear 256→256
      │                                  │
      │                                 GELU
      │                                  │
      │                                  ▼
      │                           expert representation
      │                              (256 each)
      │                                  │
      ├───────────────┐                  │
      │               │                  │
      ▼               ▼                  │
 TRAINING          EVAL / TEST           │
 w = p             w = 1[p ≥ 0.5]        │
      │               │                  │
      │               │                  │
      └───────┬───────┘                  │
              │                          │
              ▼                          │
   Zero weights for parents              │
   without experts:                      │
   MT, LY, ID                            │
              │                          │
              ▼                          │
      normalize weights                  │
          w / Σw                         │
      (Σw clamped ≥ 1e-6)                │
              │                          │
              └──────────┬───────────────┘
                         ▼
                  WEIGHTED MIXTURE
                  h_mix = Σ w_p E_p(h)
                         │
                         ▼
               Shared child classifier
                    Linear 256→16
                         │
                         ▼
                  child logits (16)
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
       parent logits             child logits
             │                       │
             └───────────┬───────────┘
                         ▼
                 25 total logits
                         │
             ┌───────────┴────────────┐
             ▼                        ▼
        TRAINING                 EVAL / TEST
             │                        │
       BCE(parent)              parents:
             │                  sigmoid ≥ 0.5
       BCE(child)                    │
             │                  children:
       parent_weight ×           sigmoid ≥ 0.5
       parent BCE               independently
             +                       │
       child_weight ×                │
       child BCE                     │
             │                       │
             ▼                       ▼
          LOSS                  PREDICTIONS
```

IMPORTANT:
- Child loss can backpropagate through soft routing
  into the parent probabilities / parent head.
- Hard routing is used only when self.training == False.
- Children are NOT masked by predicted parents.
- All 6 experts are computed for every example.
- If no expert parent has p ≥ 0.5 at inference:
      w = 0 for every expert
      h_mix = 0
      child logits = child-classifier bias
- routing_* metrics report the fraction of examples
  for which each parent exceeds the routing threshold.

This experiment implements a hierarchical multilingual text classification model based on XLM-R, in which the label space is explicitly divided into parent and child categories and parent predictions are used to control a set of parent-specific experts. Unlike a flat baseline, where XLM-R produces a single representation and directly predicts all labels independently through one classification layer, this model introduces an intermediate hierarchical decision process. The purpose is to allow the model to first learn the broader parent-level categories and then use those predictions to determine how much each parent-specific representation should contribute to the final child-level predictions.

The experiment uses the pretrained multilingual encoder TurkuNLP/web-register-classification-multilingual as its base encoder. The input text is tokenized with the XLM-R tokenizer and truncated to a maximum sequence length of 512 tokens. For each document, the final hidden representation corresponding to the first token (\<s>) is extracted from the encoder and used as the document-level representation. Because XLM-R-large produces a 1024-dimensional hidden representation, this vector serves as the common input to both the parent classification component and the parent-specific experts.

The original dataset labels are reorganized into a predefined hierarchy consisting of nine parent labels: MT, LY, SP, ID, NA, HI, IN, OP, and IP. Some parents have associated child labels, while others do not. For example, SP contains the child it, NA contains ne, sr, and nb, and IN contains en, ra, dtp, fi, and lt. Before training, the original label vectors are therefore transformed into a hierarchical representation. A parent is considered positive not only when it is explicitly annotated as positive in the original data, but also when at least one of its children is positive. This ensures that the parent labels are consistent with the child annotations. The resulting target vector contains the nine parent labels followed by the sixteen child labels, giving a total of 25 prediction targets.

The central difference from the flat baseline is introduced after obtaining the XLM-R representation. Instead of sending the same representation directly to one classifier covering all labels, the model first applies a parent classifier. The 1024-dimensional encoder representation is projected to 256 dimensions, passed through a GELU activation, and then mapped to the nine parent logits. Sigmoid is applied to these logits to obtain an independent probability for each parent. The model therefore learns which broad categories are likely to be present in the document before producing the child-level representation.

For every parent that has children, the model contains a separate expert network. Each expert receives the original 1024-dimensional XLM-R representation rather than the parent classifier representation. An expert consists of a linear transformation from 1024 to 256 dimensions, followed by GELU, dropout with a probability of 0.1, another linear transformation from 256 to 256 dimensions, and a final GELU activation. Consequently, each parent with children has its own 256-dimensional representation of the input. The purpose of these experts is to allow different parent categories to learn different transformations of the same multilingual encoder representation. For example, the representation learned by the IN expert can specialize in information useful for predicting the IN-related children, while the OP expert can specialize in information relevant to OP-related children.

The model then combines these expert representations through a mixture-of-experts mechanism. During training, the sigmoid probabilities produced by the parent classifier are used directly as soft routing weights. Thus, if the model assigns a high probability to a particular parent, the representation produced by that parent's expert contributes more strongly to the final mixture. Importantly, this routing mechanism is differentiable: the child classification loss can therefore influence the parent probabilities as well as the expert parameters. Every expert is evaluated for every example during training, but the parent probabilities determine how much each expert contributes to the resulting representation.

Parents without child labels, such as MT, LY, and ID in this particular hierarchy, do not have experts and are therefore assigned a routing weight of zero. The remaining routing weights are normalized so that their sum is one for each example. The resulting representation can be expressed conceptually as a weighted sum of the parent-specific expert representations, where the weight of each expert is determined by the corresponding predicted parent probability. This produces a single 256-dimensional mixture representation that is then passed to a shared child classifier. The child classifier maps this representation to the sixteen child logits.

The model therefore has two related prediction tasks. The first task is parent classification, which predicts the nine parent labels directly from the XLM-R representation. The second task is child classification, which predicts the sixteen child labels from the mixture of parent-specific expert representations. The total training loss is a weighted combination of the parent binary cross-entropy loss and the child binary cross-entropy loss. The two loss weights are treated as hyperparameters and are optimized with Optuna. This allows the experiment to investigate how much emphasis should be placed on learning the parent-level structure versus the final child-level classification task.

Another important distinction from a flat classifier is the routing behavior during inference. During training, the model uses soft routing because the continuous parent probabilities provide a differentiable mechanism through which the child loss can affect the parent classifier. During evaluation and testing, however, the routing becomes hard. Each parent probability is compared with a threshold of 0.5. Parents whose probability is at least 0.5 are considered active and receive a routing weight of one, while the remaining parents receive zero. These binary routing weights are then normalized across the active experts. Consequently, the child representation used during inference is constructed only from the experts associated with parents predicted to be active. This creates a clear separation between the training and inference procedures: training uses soft parent-dependent routing, whereas evaluation and testing use threshold-based hard routing.

The child predictions themselves remain multilabel predictions. After the shared child classifier produces sixteen logits, sigmoid probabilities are calculated independently for each child. A threshold of 0.5 is then used to determine whether each child is predicted as positive. Parent predictions are similarly obtained from the parent logits using the parent routing threshold. Thus, the model does not force the document to belong to exactly one parent or exactly one child. Multiple parents and multiple children can be predicted simultaneously, which is important for the multilabel nature of the task.

The experiment also includes hyperparameter optimization using Optuna. Rather than fixing the learning rate, weight decay, warmup ratio, parent loss weight, child loss weight, and gradient accumulation strategy manually, eight Optuna trials are performed. Each trial creates a fresh instance of the model and trains it using the sampled hyperparameters. The validation child macro-F1 is used as the main optimization criterion because the child-level classification task is the primary target of the experiment. Early stopping is used within each trial, and Optuna pruning can terminate trials whose validation performance is not promising. After the search, the best hyperparameter configuration is selected according to validation child macro-F1.

A new model is then initialized with the best hyperparameters and trained again. The final model is evaluated separately on the development and test sets. The experiment records parent-level, child-level, and overall macro- and micro-F1 scores, as well as weighted F1 for the complete label set. It also stores the logits, probabilities, predictions, labels, classification report, routing statistics, hierarchy definition, and selected hyperparameters. The routing statistics are particularly useful for this architecture because they show how frequently each parent expert is activated during inference. This makes it possible to examine not only classification performance but also how the learned routing mechanism is being used.

In contrast, the flat XLM-R baseline would normally take the same encoder representation and pass it directly to a single classification layer that produces all 25 label logits. Every label is therefore predicted from the same shared representation without explicitly modeling the parent-child structure. The flat model has no parent classifier that controls subsequent processing, no parent-specific experts, and no routing mechanism. A prediction for a child label is consequently not explicitly conditioned on a parent-specific representation. In the proposed hierarchical experiment, the parent predictions instead influence which specialized representations are used to construct the child representation. The model therefore introduces an additional inductive bias: labels belonging to different parent categories can be handled by different expert transformations, while the final child classifier remains shared across the hierarchy.

The main conceptual difference can therefore be summarized as follows. The flat model asks XLM-R to learn one general representation that is simultaneously useful for all labels and then predicts the labels directly. The hierarchical model first learns broad parent-level information, uses that information to route the document representation through multiple parent-specific experts, combines the resulting representations according to the predicted parent probabilities, and finally uses the resulting mixture to predict the child labels. The experiment is consequently designed to test whether explicitly incorporating the hierarchical structure of the label space and allowing different parent categories to develop specialized representations can improve child-level multilabel classification compared with a conventional flat XLM-R classifier.

An additional characteristic of this experiment is that the hierarchy is used primarily as a representation and routing mechanism rather than as a strict prediction constraint. A child prediction is not forcibly set to zero simply because its parent is predicted as negative. Instead, parent predictions determine the expert mixture, while child labels are still independently classified by the shared child classifier. This means that the model can potentially retain useful information from multiple parent experts and does not impose a rigid parent-to-child decision rule. The architecture therefore combines hierarchical supervision, parent-dependent representation learning, and mixture-of-experts routing while retaining the flexibility of multilabel prediction.
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
Parent-specific experts (Gold-Route)
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
| Flat XLM-R Baseline | Predict all 25 labels directly from a shared XLM-R representation using one classifier. | None explicitly. Parent and child labels are treated as independent multilabel targets. |
| Experiment 1: Multi-Task Hierarchical Classification | Use separate parent and child classification heads over the same XLM-R representation, with separate weighted losses. | Hierarchy as supervision. Parent and child labels are predicted separately, and the training objective explicitly distinguishes the two levels. |
| Experiment 2: Parent-Conditioned Child Classification | Learn a parent-specific representation and provide it together with the original XLM-R representation to the child classifier. | Hierarchy as information flow. Parent-level information is explicitly provided to the child prediction pathway, but there is no routing. |
| Experiment 3: Parent-Specific Expert Networks | Use a separate expert network for the children of each parent, while all experts receive the shared XLM-R representation. | Hierarchy as specialization. Each parent with children has its own expert, but all experts run for every example; parent predictions do not control routing. |
| Experiment 4: Gold-Routed Train / Hard-Routed Evaluation | Train child experts using gold parent labels, but use predicted parent labels to activate experts during evaluation. | Hierarchy as hard routing with gold training supervision. Gold parents determine expert activation during training; predicted parents determine activation during inference. |
| Experiment 5: Hard-Routed Train / Hard-Routed Evaluation | Use predicted parent labels to route examples to parent-specific child experts during both training and inference. | Hierarchy as hard conditional routing. Child prediction depends on predicted parent decisions during both training and evaluation. |
| Experiment 6: Soft-Routed Train / Hard-Routed Evaluation | During training, every expert contributes according to its predicted parent probability; during evaluation, experts are activated using a hard threshold. | Hierarchy as differentiable routing during training and hard routing during inference. Child loss can backpropagate through the soft parent probabilities. |
| Experiment 7: Parent-Specific Mixture-of-Experts | Combine parent-specific expert representations using parent probabilities during training and hard parent activation during inference, followed by a shared child classifier. | Hierarchy as representation mixture. Parent predictions determine how expert representations are combined rather than directly selecting individual child classifiers. |



---



| Model | Parent Prediction | Child Prediction | Expert Networks | Routing Strategy | Hierarchy Usage |
| --- | --- | --- | --- | --- | --- |
| Flat XLM-R Baseline | Directly predicts 9 parent labels as part of the 25-label output. | Directly predicts 16 child labels from the same shared representation. | None; one shared classifier. | No routing. | None explicitly; all 25 labels are treated independently. |
| Experiment 1: Multi-Task Hierarchical Classification | Dedicated parent classifier: XLM-R representation → 9 parent logits. | Dedicated child classifier: XLM-R representation → 16 child logits. | None; parent and child heads share the XLM-R encoder. | No routing. | Hierarchy is used as separate supervision through parent and child losses. |
| Experiment 2: Parent-Conditioned Child Classification | Parent representation is learned and used to predict 9 parent labels. | Uses the concatenation of the original XLM-R representation and the parent representation to predict 16 children. | No parent-specific experts; one shared child classifier. | No hard routing. Parent information is provided continuously to the child classifier. | Hierarchy is used to condition child prediction on learned parent-level information. |
| Experiment 3: Parent-Specific Expert Networks | Dedicated parent branch predicts 9 parent labels. | Each parent-specific expert predicts only the children belonging to that parent. | 6 separate experts for SP, NA, HI, IN, OP, and IP. | No routing; all 6 experts run for every example. | Hierarchy is used for expert specialization. Each parent has its own child representation and classifier. |
| Experiment 4: Gold-Routed Train / Hard-Routed Evaluation | Parent classifier predicts 9 parent labels. | Parent-specific experts predict their corresponding children. | 6 parent-specific experts. | **Training:** gold-parent hard routing. **Evaluation:** predicted-parent hard routing. | Hierarchy is used for hard conditional routing; gold parents provide oracle routing during training. |
| Experiment 5: Hard-Routed Train / Hard-Routed Evaluation | Parent classifier predicts 9 parent labels. | Only experts corresponding to predicted active parents produce child predictions. | 6 parent-specific experts. | **Training:** predicted-parent hard routing. **Evaluation:** predicted-parent hard routing. | Hierarchy is used as hard conditional routing throughout training and inference. |
| Experiment 6: Soft-Routed Train / Hard-Routed Evaluation | Parent classifier predicts 9 parent probabilities. | Expert child logits are weighted by the corresponding parent probability during training; active experts produce children during evaluation. | 6 parent-specific experts. | **Training:** soft routing using parent probabilities. **Evaluation:** hard routing using a threshold. | Hierarchy is used as a differentiable routing mechanism during training and a hard routing mechanism during inference. |
| Experiment 7: Parent-Specific Mixture-of-Experts | Parent classifier predicts 9 parent probabilities. | A shared child classifier predicts all 16 children from a weighted mixture of parent-specific expert representations. | 6 parent-specific experts followed by one shared child classifier. | **Training:** soft mixture using parent probabilities. **Evaluation:** hard parent activation followed by normalized mixture. | Hierarchy is used to construct a parent-dependent mixture representation for child prediction, rather than directly routing individual child classifiers. |

---

| Model | Main Strength | Main Weakness |
| --- | --- | --- |
| Flat XLM-R Baseline | Simple architecture and direct optimization of all 25 labels; provides a strong reference point. | Ignores the known parent–child hierarchy and treats all labels as independent. |
| Experiment 1: Multi-Task Hierarchical Classification | Explicitly supervises parent and child levels and allows their importance to be controlled through separate loss weights. | Parent and child predictions remain largely independent; parent information is not directly used to improve child representations. |
| Experiment 2: Parent-Conditioned Child Classification | Gives the child classifier explicit access to learned parent-level information while retaining the original XLM-R representation. | Uses a single shared child classifier, so it does not provide parent-specific specialization. |
| Experiment 3: Parent-Specific Expert Networks | Allows each parent to have a specialized expert for its children while avoiding errors caused by hard routing. | All experts run for every example, making the model computationally expensive and not explicitly conditioned on predicted parent membership. |
| Experiment 4: Gold-Routed Train / Hard-Routed Evaluation | Provides child experts with correct parent context during training, allowing them to learn without being affected by early parent-prediction errors. | Creates a train–test mismatch: training uses gold routing while inference depends on predicted parent routing; parent errors can block correct child predictions at test time. |
| Experiment 5: Hard-Routed Train / Hard-Routed Evaluation | Training and inference use the same routing mechanism, making the model behavior consistent between training and testing. | Incorrect parent predictions can prevent the correct child expert from being activated, causing error propagation from parent to child. |
| Experiment 6: Soft-Routed Train / Hard-Routed Evaluation | Soft routing allows child loss to influence the parent classifier and provides a differentiable connection between parent and child prediction. | Training and inference still use different routing mechanisms; soft logit gating can also produce unintuitive effects because probabilities scale logits rather than probabilities. |
| Experiment 7: Parent-Specific Mixture-of-Experts | Combines information from multiple parent-specific experts and allows parent predictions to influence the child representation without directly masking individual children. | More complex architecture; hard inference routing can still depend on parent errors, and the shared child classifier may reduce the specialization benefit of separate child heads. |



 ### 1\. Results table (if you have metrics)

| Model                                  | Micro-F1 | Macro-F1 | Parent Micro-F1 | Parent Macro-F1 | Child Micro-F1 | Child Macro-F1 |
|----------------------------------------|----------|----------|-----------------|-----------------|----------------|----------------|
| Flat XLM-R                             | 0.77     | 0.76     | 0.79               | 0.77               | 0.73              | 0.75              |
| Hierarchical Multitask XLM-R           | 0.77     | 0.77     | 0.80            | 0.78            | 0.73           | 0.76           |
| Parent-Conditioned Hierarchical XLM-R  | 0.76     | 0.75     | 0.78               | 0.76               | 0.73              |  0.74             |
| Parent-Specific Expert XLM-R             | 0.76     | 0.74     | 0.80            | 0.79            | 0.69           | 0.72           |
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

| Model | Short Description |
| --- | --- |
| Flat XLM-R Baseline | A flat multilabel classifier that predicts all 25 labels independently from a shared XLM-R representation. |
| Experiment 1: Multi-Task Hierarchical Classification | Separately predicts 9 parent labels and 16 child labels using two classification heads with independently weighted losses. |
| Experiment 2: Parent-Conditioned Child Classification | Uses a learned parent-level representation together with the original XLM-R representation to predict the child labels. |
| Experiment 3: Parent-Specific Expert Networks | Uses a separate expert network for each parent with children; all experts process every example and specialize in their corresponding child groups. |
| Experiment 4: Gold-Routed Train / Hard-Routed Evaluation | Uses gold parent labels to route examples to child experts during training and predicted parent labels for hard routing during evaluation. |
| Experiment 5: Hard-Routed Train / Hard-Routed Evaluation | Uses predicted parent labels to hard-route examples to parent-specific child experts during both training and evaluation. |
| Experiment 6: Soft-Routed Train / Hard-Routed Evaluation | Uses parent probabilities as soft routing weights during training, then switches to hard parent routing during evaluation. |
| Experiment 7: Parent-Specific Mixture-of-Experts | Uses parent-specific experts to produce representations that are mixed according to parent probabilities and passed to a shared child classifier. |


---

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
# Conclusion and Analysis

Overall, the results show that explicitly incorporating the label hierarchy can be beneficial, but more explicit hierarchical modeling does not necessarily lead to better performance. The Hierarchical Multitask XLM-R model achieved the best results among the proposed hierarchical approaches, with a small improvement over the flat baseline, particularly in child macro-F1. This suggests that the hierarchy provides useful additional supervision when parent and child tasks are learned jointly without making child prediction dependent on parent predictions.

The expert-based and routing approaches generally performed worse than the flat and multitask models. In particular, the results suggest that hard routing introduces error propagation: an incorrect parent prediction can prevent the corresponding child expert from being activated, making correct child prediction impossible. The gap between parent and child performance in several experiments also indicates that better parent classification does not automatically translate into better fine-grained child classification. The SoftRoute and Mixture-of-Experts results further suggest that increasing the complexity of the hierarchical mechanism does not necessarily provide a useful inductive bias for this task.

Therefore, the experiments provide evidence that the label hierarchy is useful primarily as an additional learning signal rather than as a strict routing constraint. The relatively simple multitask formulation benefits from the hierarchical structure while preserving the shared XLM-R representation and avoiding the cascading errors introduced by hard routing.

For the error analysis, the results also motivate distinguishing between two types of child errors: errors caused by an incorrect parent prediction and errors occurring even when the parent is correctly identified. This distinction can help determine whether future improvements should focus on parent-level classification or on the fine-grained discrimination between children within the same parent. Overall, the experiments indicate that exploiting the hierarchy is promising, but a lightweight hierarchical formulation is more effective for this dataset than increasingly complex expert and routing architectures.
 
---

 # Summary of Experiments (Super compact mode)


The aim was to investigate whether explicitly modeling the parent–child label hierarchy improves multilingual multilabel classification, particularly for the 16 fine-grained child labels. The task contains 9 parent and 16 child labels, with multiple labels possible at both levels.

I started with a flat XLM-R baseline and progressively introduced different forms of hierarchical modeling.

| Model | How it was performed | Overall Micro / Macro | Parent Micro / Macro | Child Micro / Macro |
|---|---|---|---|---|
| Flat XLM-R | All 25 labels were predicted directly from the shared XLM-R representation. | 0.77 / 0.76 | 0.79 / 0.77 | 0.73 / 0.75 |
| Hierarchical Multitask | Separate parent and child classifiers were trained from the shared XLM-R representation using separate losses. | 0.77 / 0.77 | 0.80 / 0.78 | 0.73 / 0.76 |
| Parent-Conditioned | Parent-level representations were provided as additional input to a shared child classifier. | 0.76 / 0.75 | 0.78 / 0.76 | 0.73 / 0.74 |
| Parent-Specific Experts | Separate experts were created for each parent with children; all experts processed each example. | 0.76 / 0.74 | 0.80 / 0.79 | 0.69 / 0.72 |
| Gold-Routed Experts | Gold parents were used to route examples during training; predicted parents were used at test time. | 0.76 / 0.75 | 0.78 / 0.76 | 0.73 / 0.74 |
| Hard-Routed Experts | Predicted parent labels were used for routing during both training and testing. | 0.73 / 0.70 | 0.75 / 0.96 | 0.70 / 0.71 |
| Soft-Routed Experts | Parent probabilities weighted the expert representations during training; hard routing was used at test time. | 0.73 / 0.66 | 0.80 / 0.79 | 0.60 / 0.58 |
| Mixture-of-Experts | Parent-specific representations were weighted by parent probabilities and combined before shared child classification. | 0.35 / 0.21 | 0.61 / 0.39 | 0.13 / 0.11 |


The Hierarchical Multitask model gave the best overall hierarchical result, improving child Macro-F1 from 0.75 to 0.76 while maintaining the same overall Micro-F1 as the flat baseline.

More complex approaches did not improve performance. Parent conditioning and expert-based models generally reduced child performance. Hard routing also showed evidence of error propagation, where an incorrect parent prediction can activate the wrong child expert.

The comparison between gold-routed and predicted-routed models suggests that routing errors contribute to this degradation. Soft routing did not resolve the problem, while the mixture-of-experts model performed substantially worse and requires further investigation.


Overall, the results suggest that the hierarchy is useful primarily as additional supervision, rather than as a mechanism for routing child predictions. The simple multitask formulation therefore appears more promising than the more complex routing architectures tested so far.

The next step is to perform error analysis to determine whether child errors mainly come from incorrect parent predictions, routing decisions, or intrinsically difficult child categories. This will help identify where further improvements to the hierarchical model are most useful.