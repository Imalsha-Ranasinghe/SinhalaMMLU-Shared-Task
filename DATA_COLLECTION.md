# Past-paper collection plan

We need more Sinhala-medium MCQs for training, at all three levels of the shared task.
You download papers by hand; a script then turns them into MCQs automatically.

| Level | Grades | Exams | Options per question |
|---|---|---|---|
| Easy | 6–8 | term test papers | 4 |
| Medium | 9–11 | term test papers, O/L past papers | 4 |
| Hard | 12–13 | A/L past papers, grade 12–13 term tests | 5 |

The Dev Set only covers Easy, so **Medium and Hard matter most**: we have no training data for them yet.

## Rules (read first)

- **Download by hand, in your browser, one paper at a time.** pastpapers.wiki allows this. Its terms forbid
  scripts, bots, download-manager extensions and "download all" tools, so don't use any.
- **Don't publish the PDFs** or the questions anywhere public (GitHub, Drive links, Hugging Face).
  They are for our own training only. Share them with the team privately (see the last section).
- Sources: [pastpapers.wiki](https://pastpapers.wiki) (main), and
  [e-Thaksalawa](https://www.e-thaksalawa.moe.gov.lk/) if it opens for you.

## Who downloads what

Two people. Each owns different levels, so nobody downloads the same paper twice.

- **Member: Medium** (grades 9, 10, 11 term tests and O/L), 20 subjects × 8 papers = **160 papers**,
  then **Easy** top-up (grades 6–8), 14 subjects × 3 papers = **42 papers**
- **Team lead: Hard** (A/L and grade 12–13 term tests), 20 subjects × 6 papers = **120 papers**, plus
  running the extraction and checking scanned papers (last section). Whoever finishes first helps with
  the other list: agree on which subjects first, so you don't both download the same ones.

Targets are minimums: more is better if you have time. Spread papers across grades, years and provinces
rather than taking 8 of the same exam.

**Folder name** = the subject name exactly as written in the tables below.

### Member: Medium (grades 9–11)

| Folder name | Target | Done |
|---|---|---|
| History | 8 | |
| Drama and Theatre | 8 | |
| Dancing | 8 | |
| Eastern Music | 8 | |
| Arts | 8 | |
| Buddhism | 8 | |
| Catholicism | 8 | |
| Christianity | 8 | |
| Islam | 8 | |
| Citizenship Education | 8 | |
| Health and Physical Science | 8 | |
| Geography | 8 | |
| Science | 8 | |
| Sinhala Language and literature | 8 | |
| Business and Accounting Studies | 8 | |
| Entrepreneurship Studies | 8 | |
| Home Economics | 8 | |
| Communication and Media Studies | 8 | |
| Agriculture and Food Technology | 8 | |
| Design and Construction Technology | 8 | |

### Team lead: Hard (A/L, grades 12–13)

| Folder name | Target | Done |
|---|---|---|
| Drama and Theatre | 6 | |
| Buddhism | 6 | |
| Christianity | 6 | |
| Islam | 6 | |
| Geography | 6 | |
| Sinhala Language and literature | 6 | |
| Business and Accounting Studies | 6 | |
| Home Economics | 6 | |
| Communication and Media Studies | 6 | |
| Agriculture and Food Technology | 6 | |
| Economics | 6 | |
| Biosystems Technology | 6 | |
| Buddhist Civilization | 6 | |
| Political Science | 6 | |
| Physics | 6 | |
| Chemistry | 6 | |
| Biology | 6 | |
| Oriental Music | 6 | |
| History of Sri Lanka | 6 | |
| Dancing Indigenous | 6 | |

### Member: Easy top-up (grades 6–8)

3 papers each for: History, Drama and Theatre, Dancing, Eastern Music, Arts, Buddhism, Catholicism,
Christianity, Islam, Citizenship Education, Health and Physical Science, Geography, Science,
Sinhala Language and literature.

## Which papers to pick

A quick check before you download, about 30 seconds per paper:

1. **Sinhala medium.** Skip Tamil and English medium.
2. **Has an MCQ section**: questions with 4 numbered options ((1) to (4)), or 5 for A/L. It's usually
   Part I. Skip papers that are only essay or structured questions.
3. **Has the answers.** Either the paper says "with Answers" (answers inside the same PDF), or you also
   download its **marking scheme** (common for O/L and A/L). Without answers, the questions are useless to us.
4. **Typed is better than scanned.** In the PDF viewer, try to select a line of text. If you can, it's typed,
   and we get the text exactly. Scanned papers still work, but they need checking by hand afterwards, so
   take a scanned one only when there is no typed alternative.
5. **Years:** prefer **2024–2026** and **before 2017** (Hard: before 2012 or 2024+). The benchmark was built
   from 2017–2023 papers (Hard: 2012–2023), so more of those get filtered out as overlap with the
   test data. They're still allowed if you can't find others.

## Saving files

Keep the **original file name** from the site: the script reads the year, grade and province from it.

```
data/raw_pdfs/
  History/
    2024-Grade-10-History-3rd-Term-Test-Paper-with-Answers-Western-Province.pdf
  Political Science/
    2015-AL-Political-Science-Past-Paper-Sinhala-Medium.pdf
    2015-AL-Political-Science-Past-Paper-Sinhala-Medium_answers.pdf
```

- **Marking scheme in a separate file:** rename it to the paper's file name + `_answers`, as above.
  This is the only renaming you should do.
- If a file name has no grade, year or level in it (e.g. `paper.pdf`), rename it to the same pattern:
  `<year>-Grade-<NN>-<Subject>-...pdf`, or `<year>-OL-<Subject>-...pdf` / `<year>-AL-<Subject>-...pdf`.
- Delete duplicate downloads such as `... (1).pdf`.
- Optional, but useful for the report: add a line to `data/raw_pdfs/sources.csv` with the file name and
  the page you downloaded it from:
  ```
  file,url
  2024-Grade-10-History-3rd-Term-Test-Paper-with-Answers-Western-Province.pdf,https://pastpapers.wiki/...
  ```

## Handing in

PDFs are not committed to git (`data/raw_pdfs/` is ignored). Zip your subject folders and share the zip
privately with the team lead (Drive shared only with the team, not "anyone with the link").
Fill in the **Done** column above, or just tell the lead your counts.

## For the team lead: turning PDFs into MCQs

Unzip everything into `data/raw_pdfs/`, then:

```bash
python -m src.papers data/raw_pdfs --limit 2     # try two papers, read data/papers/json/*.json
python -m src.papers data/raw_pdfs               # everything; rerun to resume after a quota stop
```

- Output: `data/papers/json/<Subject>_papers.json` in the Dev Set format, with `metadata.difficulty` set
  from the grade. Load it with `src.data.load_raw("data/papers/json")`.
- `data/papers/report.json` lists per-paper counts and why questions were dropped. A paper with
  `answers_in_key: 0` lost its answers, so check its `_answers` file name.
- Read a sample of the questions from scanned papers (`metadata.text_layer: false`) before training.
- Run it every day or two as zips come in: finished papers are cached, so only new ones cost API calls.
- To also filter against the public SinhalaMMLU release, pass it with `--decontam <file.json>`.
