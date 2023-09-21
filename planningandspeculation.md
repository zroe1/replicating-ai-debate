Paper structure:

Abstract

Introduction
- Motivation of original experiment and for our replication
- Importance of replication in general AI research
Intention: have models train via debate to eliminate deceptive strategies

Methods

Results

Discussion
- Results of replication
- Application, scalability, moving forward 
///

Speculation

Pre-commit is a reasonable restriction on agents since an agent would have no reason to change their claim based on information raised by an opponent (? Or could an agent change their mind due to opponent reasoning? But the two would presumably have access to the same information)

Scalability
The application of the results of this experiment relies on a few key assumptions. Firstly, that a context in which the judge has limited knowledge is analogous to one in which the judge has inferior intelligence to the two debating agents. Second, the experiment uses a literal image as a stand-in for a metaphorical bigger picture where pixels are analogous to components within the bigger picture.

Limited number of components shown could also present an issue of misrepresentation. If say 90% of evidence supports one claim and 10% supports another but agents are only able to reveal a very very small percentage of the total evidence/ components of the total truth, the two claims may appear equally strong even though the big picture may look very different. Then it would be a matter of having the revealed information be a sufficient percentage of the total information; however, this might be difficult for very large-scale issues, especially when considering agents far more intelligent than humans.

Claim that it is easier to tell the truth than to get away with a lie. But what about a human judge? We may not be able to train a ML judge on all tasks (example?) Human judge could be susceptible to deceptive behavior due to biases that AI could exploit.

Interesting that some lies are easier than others. Liar agent is randomly assigned a lie but real agents in training may learn to lie more effectively aka choose lies that appear close to the truth.
