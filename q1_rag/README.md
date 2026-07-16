# Homework 3, Part 1.1

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
```

## Run

```bash
python demo.py --knowledge-base cat-facts.txt --top-n 3
python demo.py --knowledge-base cat-db-fiction-confusing.txt --top-n 3
```

The program loads one non-empty line per chunk, embeds all chunks, computes cosine similarity without a vector database, prints the retrieved evidence, and then streams a grounded answer.

## Evaluation table





<table>
  <thead>
    <tr>
      <th>File used</th>
      <th>Question no.</th>
      <th>top_n</th>
      <th>Generated answer</th>
      <th>Correct?</th>
      <th>If incorrect, possible fix</th>
    </tr>
  </thead>

  <tbody>
    <tr>
      <td><code>cat-facts.txt</code></td>
      <td>1</td>
      <td>3</td>
      <td>
        <strong>Retrieved knowledge:</strong><br>
        • (0.862) Besides smelling with their nose, cats can smell with an additional organ called the Jacobson’s organ, located in the upper surface of the mouth.<br>
        • (0.666) Studies now show that the allergen in cats is related to their scent glands.<br>
        • (0.652) A cat rubs against people to mark its territory with scent glands.<br><br>

        <strong>Answer:</strong><br>
        The additional organ that allows cats to smell besides their nose is the Jacobson's organ, which is located in the upper surface of the mouth. Specifically, it is found on either side of the jawbone.
      </td>
      <td>Yes</td>
      <td>–</td>
    </tr>

    <tr>
      <td><code>cat-facts.txt</code></td>
      <td>2</td>
      <td>3</td>
      <td>
        <strong>Retrieved knowledge:</strong><br>
        • (0.832) On average, cats spend 2/3 of every day sleeping. That means a nine-year-old cat has been awake for only three years of its life.<br>
        • (0.747) The average life span of a domestic cat is 14 years.<br>
        • (0.718) The oldest cat on record lived for more than 38 years.<br><br>

        <strong>Answer:</strong><br>
        Since a cat spends 2/3 of every day sleeping, and assuming an average waking period of around 22 hours per day, we can estimate the total number of days spent awake.<br><br>

        The model then performed several incorrect conversions and concluded that the cat had been awake for approximately three or four years.
      </td>
      <td>No</td>
      <td>
        Instruct the generator to avoid unnecessary unit conversions and use the proportion directly:
        if the cat sleeps 2/3 of the time, it is awake 1/3 of the time, so 9 × 1/3 = 3 years.
        Alternatively, use a calculator tool for arithmetic.
      </td>
    </tr>

    <tr>
      <td><code>cat-db-fiction-confusing.txt</code></td>
      <td>3</td>
      <td>3</td>
      <td>
        <strong>Retrieved knowledge:</strong><br>
        • (0.695) The Riverside Luna must eat rabbit-based food because her record lists a chicken allergy.<br>
        • (0.652) Luna at Northside wears a green collar, while Luna at Riverside wears a yellow collar.<br>
        • (0.591) The Northside Luna eats standard chicken-based food.<br><br>

        <strong>Answer:</strong><br>
        The Riverside Luna must eat rabbit-based food because her record lists a chicken allergy. She wears a yellow collar.
      </td>
      <td>Yes</td>
      <td>–</td>
    </tr>

    <tr>
      <td><code>cat-db-fiction-confusing.txt</code></td>
      <td>4</td>
      <td>3</td>
      <td>
        <strong>Retrieved knowledge:</strong><br>
        • (0.862) Juniper was scheduled for the Saturday adoption event on 2026-05-02.<br>
        • (0.702) Juniper's sister Jupiter may attend the event and has no activity restriction.<br>
        • (0.661) A veterinary note dated 2026-05-01 says Juniper must rest for seven days and cannot attend adoption events during that period.<br><br>

        <strong>Answer:</strong><br>
        The model stated that Juniper might still be able to attend the event, despite acknowledging that she was prohibited from attending adoption events during the seven-day rest period.
      </td>
      <td>No</td>
      <td>
        Instruct the model to prioritize newer dated evidence and explicitly compare the event date with the restriction period.
        The veterinary note from 2026-05-01 overrides the earlier schedule, so Juniper cannot attend on 2026-05-02.
      </td>
    </tr>
  </tbody>
</table>
