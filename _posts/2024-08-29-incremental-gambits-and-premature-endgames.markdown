---
layout: post
title:  "Incremental Gambits and Premature Endgames"
date:   2024-08-29
categories: blog
---

In chess, there are three phases to the game, all with distinct strategies: the opening, the middlegame, and the endgame. In the opening, the first 15 or so moves of the game, the goal is to deploy your pieces from their starting positions and control the center of the board. The middlegame involves back-and-forth offense and defense as players target each other's pieces via tactics and attempt to checkmate the king. The endgame begins when most pieces have been captured without a checkmate. Only a few pieces remain, and a slower, mathematical trudge begins to checkmate the king and declare victory.

Plenty of cliche parallels exist between chess and business, but nowhere do I find them more relevant than in the current AI race. The progression has been:

- **Opening (2017-2022):** Researchers and companies developed models based on the transformer architecture, setting the pieces on the board.  
- **Middlegame (2022-2024):** ChatGPT blew the center wide open. It sparked rapid-fire iteration and fierce competition, with companies racing to deploy applications built on top of LLMs.  
- **Endgame (Today):** A handful of dominant players—chip designers, data center suppliers, hyperscalers, and top model houses—have consolidated power and are fighting to maintain their leads.

The status quo assumes that the transformer architecture will win. Companies up and down the stack have coalesced around transformers thanks to their generalizability and scalability. Consolidation has led to an innovation landscape dominated by incremental gambits – clever little hacks meant to squeeze more juice out of models without fundamentally advancing their intelligence.

We've prematurely entered the endgame of the AI wave, crowning a winning technology before getting the chance to deploy all of our pieces. This essay will examine the limitations of our current approach and propose alternatives that may lead us closer to superintelligence. As it stands, we risk a stalemate – trapped in a local optimum lacking the fundamentals to unlock AGI.

![Chess](/images/Chess.png)

### The Old King

[*Attention is All You Need*](https://arxiv.org/abs/1706.03762) introduced the [transformer](https://poloclub.github.io/transformer-explainer/) in 2017\. Its [self-attention mechanism](https://www.sciencedirect.com/topics/computer-science/self-attention-mechanism\#:\~:text=A%20Self%2DAttention%20Mechanism%20is,when%20making%20predictions%20or%20decisions.) allows the model to capture contextual relationships between words more efficiently. In 2018, Google released [BERT](https://arxiv.org/abs/1810.04805), which introduced bi-directional pre-training. A massive, plain text dataset was used to train the model before fine-tuning it to a specific task. This helped influence the development of OpenAI's GPT models, which were trained on large corpora of text (and later other [multi-modal data](https://www.theverge.com/2024/4/6/24122915/openai-youtube-transcripts-gpt-4-training-data-google)) and demonstrated impressive zero-shot learning capabilities. When ChatGPT was released in November 2022, the transformer went mainstream and started the race to build around it.

These early successes paved the way for the ubiquity of transformers across AI. The transformer's ability to [scale](https://arxiv.org/abs/2001.08361) effectively, capture long-range dependencies, and adapt to various language-based tasks has made them the go-to architecture across textual domains, including less obvious areas like [biology](https://www.nature.com/articles/s41592-024-02354-y) and [chemistry](https://arxiv.org/abs/2408.07246). Today, companies like OpenAI (GPT), Meta (Llama), Anthropic (Claude), xAI (Grok), and more are all-in on transformers and pushing the boundaries of performance.

Despite the success, transformers feel like they're starting to reach their limits. While models incrementally improve with each iteration, it has become apparent that the path to AGI is [far from clear](https://www.pcmag.com/news/meta-ai-chief-large-language-models-wont-achieve-agi) using transformers alone. LLM performance gains are becoming increasingly [difficult](https://dl.acm.org/doi/abs/10.1145/3531146.3533229) to achieve due to their insatiable [appetite](https://arxiv.org/abs/2404.04125) for training data and compute. The most advanced models [struggle](https://arxiv.org/abs/2210.09261) with complex reasoning tasks that require multi-step logical inference or abstract thinking. LLMs often [fail](https://arxiv.org/abs/2306.03341) to generalize knowledge in novel ways, relying on pattern-matching within their training data rather than demonstrating proper cognitive understanding. Realistically, [superintelligence](https://situational-awareness.ai/) is unlikely to emerge out of this architecture.

While transformers have completely upended consumer and enterprise workflows, they won't be the final step toward AGI (and not even [SOTA within a few short years](https://www.isattentionallyouneed.com/)). The utopian hype following the release of ChatGPT has waned as expectations inflate. I'll explore these challenges in the following sections, examining why transformers fall short in crucial areas like reasoning, generalization, and efficient scaling. We'll also look at other approaches that address these fundamental issues and offer alternative endgames on the march toward AGI.

### Incremental Gambits

At the heart of the transformer lies its fundamental challenge (ironically caused by its greatest strength): the [quadratic computational complexity](https://www.sciencedirect.com/topics/computer-science/quadratic-time-complexity\#:\~:text='Quadratic%20Time%20Complexity'%20refers%20to,techniques%20like%20RLTP%20and%20LTrP.) problem caused by self-attention. This alone will prevent the transformer architecture from enabling AGI.

In a transformer, each token in a sequence must attend to every other token, resulting in a computational complexity that grows quadratically with input length. In other words, a sequence of length *n* results in *n^2* computations. This means that doubling the length of an input doesn't just double the computational requirements — it quadruples them. This kind of scaling poses a problem for training and inference, especially as we aim to process longer text sequences or incorporate multiple modalities. I won't go into deeper detail about the mechanics, but you can find a tremendous technical writeup on self-attention [here](https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html).

![Self-Attention](/images/self-attention.png)
[*Explainable AIL Visualizing Attention in Transformers (Comet)*](https://www.comet.com/site/blog/explainable-ai-for-transformers/)

The impact of quadratic complexity is evident in the limited context windows of current commercial models. For instance, despite having an [estimated](https://the-decoder.com/gpt-4-architecture-datasets-costs-and-more-leaked/) 1.7B+ parameters, GPT-4o has a [context window of just 128K tokens](https://platform.openai.com/docs/models/gpt-4o) — equivalent to about 300 pages of text. Quadratic complexity has spawned a wave of research aimed at solving it.

A gambit in chess is when a player sacrifices a piece during the opening in exchange for material compensation later. They are tricks meant to do more with less. I call these recent improvements to manage quadratic complexity incremental gambits because they offer marginal advantages over base transformers but often introduce new trade-offs while remaining inherently based on neural nets.

One popular approach has been the development of [sparse attention mechanisms](https://medium.com/@vishal09vns/sparse-attention-dad17691478c). Models like [Longformer](https://arxiv.org/abs/2004.05150) use patterns of sparse attention to reduce computational complexity by focusing on relationships between only a strategic subset of relevant tokens. While this allows for the efficient processing of much longer sequences, it can potentially miss critical long-range dependencies in data. Another tactic has been the exploration of [linear attention mechanisms](https://arxiv.org/abs/2102.11174). These have shown that it's possible to approximate the attention computation with linear operations. In turn, computational requirements are significantly reduced, but so is the expressive power of the model. Researchers have also experimented with [recurrent memory techniques](https://arxiv.org/abs/2203.08913) to extend the effective context of transformers. These allow the model to access a sizable external memory, enabling theoretically indefinite context length. However, this approach adds complexity without addressing the underlying transformer scaling issues.

This [piece](https://www.mackenziemorehead.com/is-attention-all-you-need/) from Mackenzie Morehead offers a more detailed look at attention alternatives and helped inspire this section.

These gambits help make transformers (and neural nets broadly) better, but won’t actually get us closer to AGI. Through the lens of these approaches, the goal has shifted from making AGI to making transformers work, a divergence from OpenAI’s own [charter](https://openai.com/charter/). The question then becomes: “Why do we need to make transformers / neural nets work?”. I believe it’s because we’re in too deep already.

### Premature Endgames

[Chinchilla scaling laws](https://arxiv.org/abs/2203.15556) refined our understanding of the optimal balance between model size and training data. The DeepMind research team found that many LLMs were undertrained relative to their size, suggesting that making models bigger wasn't the most efficient path forward. This assertion starkly contrasted with the prevailing notions outlined in [Kaplan's scaling laws](https://arxiv.org/abs/2001.08361). As models grow, performance improvements become [increasingly unpredictable](https://arxiv.org/abs/2202.07785) and potentially disappointing, which explains why larger models have been perceived as underwhelming.

The transition from Llama-3 70B to Llama-3.1 405B illustrated this challenge well. Despite a nearly 6x increase in parameters, Llama-3.1 405B only scored [3.2 points better](https://context.ai/compare/llama3-70b-instruct-v1/llama3-1-405b-instruct-v1) than Llama-3 70B on the 5-shot MMLU benchmark. This nominal improvement relative to the massive increase in model size, computational resources, and training time signals diminishing returns and raises serious questions about the viability of continued scaling as a path to AGI.

From a purely economic perspective, the trend has translated into a staggering [capex problem](https://www.sequoiacap.com/article/ais-600b-question/) that analysts expect will worsen, even with a [delayed Blackwell](https://www.theinformation.com/articles/nvidias-new-ai-chip-is-delayed-impacting-microsoft-google-meta) GPU. The AI arms race has led to unprecedented investment in compute infrastructure, with companies betting on the assumption that larger models will justify the costs. However, the ROI remains highly questionable.

Private market arguments on this topic can be overly optimistic. Most concede that there is a demand issue but conclude that it will catch up and make investors whole. Some investment may be recouped, but this is a classic VC take that fails to capture the intricacies of how we'll get there.

In a recent episode of [BG2](https://youtu.be/5LYZCoDysLs?si=ZZ9vG7QOF42hqrW9\&t=1840), Brad Gerstner and Bill Gurley offered a more pragmatic view through the lens of public markets.

There's a growing concern that supply is outpacing demand. While Jensen [estimates that $2T](https://www.datacenterdynamics.com/en/news/nvidia-ceo-jensen-huang-predicts-data-center-spend-will-double-to-2-trillion/) will need to be spent  on data center build-out by 2028, it's unclear if end-user demand will materialize to justify the massive capital outlay. At a [projected](https://x.com/modestproposal1/status/1828456896811909531) 14% of market-wide capital spending by 2026, NVIDIA is expected to be "comparable to IBM at the peak of the Mainframe Era or Cisco, Lucent, and Nortel at the peak of the New Economy Era." A foreboding interpretation of the tea leaves as [NVIDIA](https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-second-quarter-fiscal-2025) slumps on an earnings beat as customers and investors realize it could be [mortal](https://x.com/DavidCahn6/status/1828916642480603594) after all...

Even consumer-facing AI applications are struggling to generate revenues commensurate with the hype. OpenAI expects to [bring in $3.4B in 2024](https://www.bloomberg.com/news/articles/2024-06-12/openai-doubles-annualized-revenue-to-3-4-billion-information), a fraction of what's needed to close the expanding air gap. There's a noticeable disconnect between growing user bases and actual engagement metrics, raising questions about the long-term value proposition of these offerings.

There's a misalignment between massive capex and uncertain future revenues. While big tech executives acknowledge that capex and revenues will never be perfectly aligned, it's concerning that Microsoft's CFO expects its quickly depreciating generative AI assets to [monetize over 15 years](https://www.forbes.com/sites/petercohan/2024/07/31/microsoft-stock-drops-as-ai-capital-expenditures-surge-to-56-billion/). We're in uncharted territory, hoping breakthrough applications that meaning contribute to clawing back capex will eventually [justify](https://matthewlewis.xyz/blog/2024/07/08/deep-tech-diss-track.html) these investments.

On [Invest Like the Best](https://open.spotify.com/episode/05Fx7yNSEA148kHr1znbrb?si=6c5248ab1a5045e2), Gavin Baker painted a more concerning picture. He argues that hyperscaler CEOs are in a race to create a Digital God and shares Larry Page has said internally at Google that he's "willing to go bankrupt rather than lose this race." Companies like Google will keep spending until something tells them that scaling laws are slowing, and according to them, scaling laws have only slowed because NVIDIA chips haven't improved since GPT-4 was released, and so one can deduce that capex will continue to grow... We don't know if transformers will scale to the point of AGI, but in the minds of large companies, they better because that's the only way any of this will end well financially. Sam was not tongue-in-cheek when he [said](https://www.youtube.com/watch?v=TzcJlKg2Rc0\&t=1886s) that OpenAI's business model was to create AGI and then ask how to return investor capital.

The premature rush to the AI endgame poses significant risks. We are massively over-indexing on an approach that shows diminishing returns, potentially at the cost of exploring more innovative and efficient paths to advanced reasoning. I am not saying that AI on its current trajectory isn't insanely valuable, nor am I saying that the next lift provided by new chips will accelerate progress. I will even concede that if transformers were not the architecture of the future, then this isn't all for naught as data centers can be repurposed. But this next shift will be similar to those of yesteryear. We may be over our skis. This cannot be the path to AGI.

### A New Queen

Transformers excel at recognizing and reproducing patterns they've memorized from training data but struggle with tasks that require genuine reasoning. More training data does lead to better results on memorization-based benchmarks but offers only the illusion of general intelligence. Given the black box, it's unclear whether more data improves reasoning or simply offers more potential patterns to be uncovered across contexts.

François Chollet offered a new [benchmark](https://arxiv.org/abs/1911.01547) in 2019 (formalized via a [competition](https://arcprize.org/) this summer) that shot to the forefront of the AI zeitgeist called Abstraction and Reasoning Corpus for Artificial General Intelligence, or ARC-AGI. In the prize [announcement](https://arcprize.org/blog/launch), the team states that scale will not enable LLMs to learn new skills. Instead, new generalizable architectures are needed to adapt to novel situations. ARC is designed to test an AI system's ability to understand abstract patterns and apply that understanding to novel situations while explicitly resisting memorization, the core strength of transformers. Think of it like an IQ test, testing the purest form of “intelligence,” but for machines. The SOTA [score](https://www.kaggle.com/competitions/arc-prize-2024/leaderboard) to date on ARC is 46% from the MindsAI team led by [Jack Cole](https://x.com/mindsai\_jack). An 85% is needed to claim the $500K grand prize.

While we don't know the methods of those atop the current leaderboard, I want to propose an intriguing potential solution: neurosymbolic AI (NSAI). This approach combines the best parts of self-supervised deep learning (pattern recognition and large-scale data retrieval) with symbolic approaches (explicit knowledge representation and logical reasoning), each making up for the other's [shortcomings](https://www.youtube.com/watch?v=eHWZYURQvGw\&t=957s).

A great comparison is that of Daniel Kahneman's ["Thinking Fast and Slow,"](https://kahneman.scholar.princeton.edu/publications) where neural nets represent system 1 thinking and symbolic AI system 2\. From [DeepMind](https://deepmind.google/discover/blog/alphageometry-an-olympiad-level-ai-system-for-geometry/):

Because language models excel at identifying general patterns and relationships in data, they can quickly predict potentially helpful constructs but cannot reason rigorously or explain their decisions. Symbolic deduction engines, on the other hand, are based on formal logic and use clear rules to arrive at conclusions. They are rational and explainable but can be slow and inflexible — especially when dealing with complex problems independently.

Neural nets and symbolic systems [complement](https://www.semantic-web-journal.net/system/files/swj2291.pdf) one another. While the former is robust against noise and excels at pattern recognition, symbolic systems thrive at tasks involving structured data and explicit reasoning. The challenge is reconciling the fundamentally different ways these systems represent information.

NSAI attempts to address this challenge by creating an integration that can leverage the strengths of both systems. Per [Sheth et al.](https://arxiv.org/abs/2305.00813), the integration of these can be broadly categorized across two main components:

1. A neural component based on deep learning architectures can process raw, unstructured data and perform pattern recognition.  
2. A symbolic component that can handle knowledge representation, logical reasoning, and the manipulation of abstract concepts.

The team goes on to share that the actual integration can then be achieved through two approaches:

1. Knowledge compression for neural integration involves techniques like knowledge graph embedding and logic-based compression.  
2. Neural pattern lifting for symbolic integration includes methods like decoupled and intertwined integration.

![New Hope](/images/new-hope.png)
[*Neurosymbolic AI - Why, What, and How (Sheth et al.)*](https://arxiv.org/abs/2305.00813)

When compared to mainstream LLM architectures, NSAI offers manifold [advantages](https://emeritus.org/in/learn/neurosymbolic-ai/). First is its enhanced reasoning and generalization capabilities. By incorporating oft-dismissed symbolic reasoning, these systems can generalize from fewer examples and apply knowledge more flexibly to new situations, precisely the kind of things ARC is testing for and where transformers fall short.

As a result, NSAI systems may require less training data to achieve higher performance, addressing one of the transformer’s fundamental limitations. This could democratize development, making it accessible to domains where large datasets are scarce or impossible to obtain. (Check out [DisTrO](https://github.com/NousResearch/DisTrO/blob/main/A\_Preliminary\_Report\_on\_DisTrO.pdf) from Nous Research for a snapshot of the future of more efficient LLM training.)

Unlike transformers' black boxes, NSAI outputs are more transparent, offering a sort of audit trail for each step in the process. This transparency is critical for applications in nuanced fields where understanding the "why" is as important as the decision itself.

Finally, by incorporating neural nets into the winning architecture, we can lean on the progress made over the past few years and build on top of what we have already. Human-level intelligence will use neural nets – just not rely on them in their entirety.

Looking forward, the potential applications of NSAI are vast, and integrating neural and symbolic approaches promises to compensate for the lost time and money (again, for some, not all) caused by transformers on their own. However, realizing this potential will require sustained research effort and funding.

Luckily, we're finally beginning to see NSAI approaches hit the mainstream. In January, DeepMind released [AlphaGeometry](https://deepmind.google/discover/blog/alphageometry-an-olympiad-level-ai-system-for-geometry/), which can solve complex geometry problems at an International Mathematical Olympic Gold Medalist level. Its language model guides a symbolic deduction method towards the possible range of solutions. More recently, a company called [Symbolica](https://www.symbolica.ai/) [launched](https://venturebeat.com/ai/move-over-deep-learning-symbolicas-structured-approach-could-transform-ai/) with $33M in funding. Symbolica is attempting to create a reasoning model via an approach rooted in [category theory](https://arxiv.org/abs/2402.15332), which relates mathematical structures to one another. If you know anybody else making the emergence and adoption of NSAI their mission, please reach out.

<br />
---

We've reached a juncture in the AI wave, and it's clear that the endgame has been called prematurely. Pieces have been sacrificed for perceived advantages as the industry has gone all-in on transformers and continues to blitzscale to beat the competition.

The transformer architecture (for all its impressive capabilities and economic importance) has fundamental flaws. The quadratic complexity problem has led to a series of incremental gambits that ultimately fail to solve the unsolvable. Spending in pursuit of the impossibly distant AGI has swelled to concerning levels. The disconnect between the numbers and the commentary raises serious questions about the long-term viability of this strategy.

Neurosymbolic AI does offer hope. By combining neural networks' pattern recognition strengths with symbolic systems' explicit reasoning capabilities, NSAI promises a more balanced and potentially fruitful approach to AGI. It's still a pawn today, but as it marches down the board, it will soon become a queen to aid our beleaguered king.

The most successful chess players can adapt their strategy as the game evolves and see beyond the immediate tactics to the larger strategic picture. The same is true in AI research and AI itself. We must be willing to reconsider our opening moves, develop new strategies for the middlegame, and redefine what victory looks like in the endgame.

<br />
---

### Additional Resources

There is so much more I could write on this topic. Check out the links throughout my write-up for frequently used resources. Below is a non-exhaustive list of additional papers that can help with technical context.

- [DreamCoder: Growing generalizable, interpretable knowledge with wake-sleep Bayesian program learning (Ellis et al.)](https://arxiv.org/abs/2006.08381)  
- [Natural Language Processing and Neurosymbolic AI: The Role of Neural Networks with Knowledge-Guided Symbolic Approaches (Barnes, Hutson)](https://digitalcommons.lindenwood.edu/cgi/viewcontent.cgi?article=1610\&context=faculty-research-papers)  
- [Neuro-Symbolic AI: An Emerging Class of AI Workloads and their Characterization (Susskind et al)](https://arxiv.org/abs/2109.06133)  
- [Neuro-symbolic approaches in artificial intelligence (Hitzler et al.)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9166567/)  
- [Neurosymbolic Programming (Chaudhuri et al.)](https://www.cs.utexas.edu/\~swarat/pubs/PGL-049-Plain.pdf)

<br />
---

### Acknowledgments

*\~ Thank you to [Rohan Pujara](https://x.com/rohanpuj) for helping edit*

<div align="center">###</div>