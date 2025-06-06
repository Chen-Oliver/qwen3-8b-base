# Qwen3-8B-Base

A simple from scratch PyTorch implementation of the `Qwen/Qwen3-8B-Base` model, using pre-trained weights from [Hugging Face](https://huggingface.co/Qwen/Qwen3-8B).

## Getting Started

### Prerequisites

1.  **Dependencies:** Install the required packages using `uv`.
    ```sh
    uv sync
    ```

2.  **Model Weights:** You'll need the `config.json` and the model weights for `Qwen/Qwen3-8B-Base`. You can download them from the [Hugging Face Hub](https://huggingface.co/Qwen/Qwen3-8B-Base).
    - Place `config.json` in the project root. There is already one included in this repo.
    - Download and place the model weight files (e.g., `.safetensors` and `model.safetensors.index.json`) into a `qwen3_weights/` directory.

### Running the Example

The `model.py` includes a simple demonstration when ran as a script
```sh
uv run model.py
```
This will run both greedy and sampling-based generation and print the output to the console.

## Architecture
- The Qwen3 line of models contains both Dense and Mixture of Expert Models. All models in this line utilize Grouped Query Attention.
- Qwen3-8B, like its smaller siblings (0.6B, 1.7B, 4B) is a dense model but one difference is that it doesn't utilize tie embedding.
- Qwen3 differs from Qwen2 in that it no longer uses a QKV bias and opts for a QK normalization step before applying attention. This is similar to the QK-norm used in [OlMo2](https://arxiv.org/pdf/2501.00656) but the norm is shared across the heads.


## Known Limitations
This is a simplified implementation and lacks many optimizations.
-  No batch support: single-prompt inference only
-  No KV Cache
-  No RoPE scaling
-  Precision differences
    - The [Huggingface implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py#L51) uses a custom RMSNorm kernel that leads to slight precision differences with this implementation.

## Example Generations

The model produce varied outputs based on the generation strategy.

### Greedy Generation (temperature=0.0)
- **Prompt:** `Top 3 Michelin Restaurants:`
- **Output:**
```
 A Culinary Journey Through the Stars

Michelin stars are the ultimate accolade in the world of fine dining, representing excellence in cuisine, service, and overall dining experience. These prestigious awards are given by the Michelin Guide, a renowned publication that has been evaluating restaurants since 1900. In this article, we will explore the top 3 Michelin restaurants that have earned the coveted three-star rating, offering a culinary journey through the stars.

1. El Celler de Can Roca (Spain)

Located in Girona, Spain, El Celler de Can Roca is a three-Michelin-starred restaurant that has been at the forefront of modern Spanish cuisine. The restaurant is run by the Roca brothers, Joan, Josep, and Jordi, who have been recognized for their innovative approach to cooking and their ability to create dishes that are both visually stunning and delicious.

The menu at El Celler de Can Roca is a true reflection of the brothers' creativity and passion for food. The dishes are carefully crafted to showcase the best of Spanish ingredients, with a focus on seasonality and sustainability. The restaurant's tasting menu is a journey through the flavors of Spain, with each course offering a unique and memorable experience.

2. Noma (Denmark)

Noma, located in Copenhagen, Denmark, is another three-Michelin-starred restaurant that has gained international recognition for its innovative approach to Nordic cuisine. The restaurant was founded by chef René Redzepi, who has been at the forefront of the Nordic food movement, which emphasizes the use of local and seasonal ingredients.

The menu at Noma is a celebration of Nordic flavors, with dishes that are both simple and complex. The restaurant's tasting menu is a journey through the flavors of the Nordic region, with each course offering a unique and memorable experience. The restaurant's commitment to sustainability and its focus on local ingredients have made it a leader in the culinary world.

3. Osteria Francescana (Italy)

Osteria Francescana, located in Modena, Italy, is a three-Michelin-starred restaurant that has been recognized for its innovative approach to Italian cuisine. The restaurant is run by chef Massimo Bottura, who has been at the forefront of the Italian food movement, which emphasizes the use of traditional ingredients and techniques.

The menu at Osteria Francescana is a celebration of Italian flavors, with dishes that are both simple and complex. The restaurant's tasting menu is a journey through the flavors of Italy, with each course offering a unique and memorable experience. The restaurant's commitment to sustainability and its focus on traditional ingredients have made it a leader in the culinary world.

Conclusion

The top 3 Michelin restaurants offer a culinary journey through the stars, with each restaurant offering a unique and memorable experience. Whether you are a foodie or simply looking for a special dining experience, these restaurants are a must-visit for anyone who appreciates the art of fine dining.
```

### Sampling (temperature=0.6)
- **Prompt:** `What is the capital of USA?`
- **Output:**
```
 A. india B. canada C. china D. washington, DC
D. washington, DC
Original conversation
User: What is the capital of USA? A. india B. canada C. china D. washington, DC

Weegy: D. washington, DC

User: thanks

Weegy: Your welcome. Do you have another question?

User: yes

Weegy: (More)

Question
Asked 5/13/2012 8:27:25 AM
Popular Conversations
factorization of 10x - 3 - 3x 2? User: n 4 - 1
Weegy: (n + 8)(n - 2) = n^2 - 2n + 8n - 16; = n^2 + 6n - 16 User: Factor a 3 - 3 + 3a 2 - a. User: Factor 75t 2 ...
6/30/2016 12:45:49 AM| 3 Answers
Solve (x + 1 -3). User: Find the distance between the points (-4, ...
Weegy: Distance between the points (4, -1) and (-5, 2) = sqrt [(-5-4)^2 +(2- 1)^2] = sqrt [81+9] = sqrt[90] = 9.4868 ...
6/30/2016 7:13:59 AM| 2 Answers
Bacteria are important in sewage disposal because they _____. A. ...
Weegy: d. form membranes around toxins in sewage User: Organisms that grow in the absence of free oxygen are known as ...
6/30/2016 10:07:46 AM| 2 Answers
What effect did the oil crisis have on U.S. energy policy? A. The ...

What is the capital of USA?

Weegy: B. More attention has since been paid to energy conservation. was an effect of the oil crisis User: Which ...
6/30/2016 10:13:02 AM| 2 Answers
What is the slope of the line passing through (1, 2) and (3, 8)? ...
Weegy: The slope of a line perpendicular to the line whose equation is y = 2x + 5 is -1/2. The slope of that line is ...
6/30/2016 2:59:15 PM| 2 Answers
Factor x^10 - y^2.
6/30/2016 12:29:14 AM| 1 Answers
Weegy Stuff
```

## References

- [Qwen3 Technical Report](https://arxiv.org/pdf/2505.09388v1)
- [Qwen3 Blog](https://qwenlm.github.io/blog/qwen3/)
- [HF Transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py)
