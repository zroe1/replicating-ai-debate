# Replicating "<a href="https://arxiv.org/abs/1805.00899">AI safety via debate</a>"

This repository is dedicated to replicating the results of the paper "AI safety via debate" by Geoffrey Irving, Paul Christiano, and Dario Amodei (all of OpenAI). The paper was published on October 22, 2018 and can be found <a href="https://arxiv.org/abs/1805.00899">here</a>.

### Six pixel judge model:

<b>Name: </b>six_pxl_model_NO_REPEAT_Xs.h5  
<b>Accuracy: </b> 58.3%  
<b>Notes: </b>The model was trained with 30,000 batches of 128 samples with Adam (learning rate = 10<sup>-4</sup>) as was done in the original paper. While the model in the paper had an accuracy of 59.4%, the model in this repository only achieves 58.3%. This could be for one of two reasons:  

<ol>
  <li>The paper includes a link to TensorFlow code of the model architecture. Sadly the link is to a page that no longer exists. As a result, this repository uses a <a href="https://www.tensorflow.org/tutorials/quickstart/advanced">model architecture on a different page on the TensorFlow website</a>. Because it is unclear whether they are the same or similar models, it is hard to tell if this is affecting performance.</li>
  <li>The paper is (in my reading of it) unclear about how to prepare the input to the model. I used my best guess based on my reading of the paper, but my implementation could be negatively affecting performance.</li>
</ol>
 
### Four pixel judge model:

<b>Name: </b> four_pxl_judge_NO_REPEAT_Xs.h5  
<b>Accuracy: </b> 46.9%  
<b>Notes: </b>The model was trained with 50,000 batches of 128 samples with Adam (learning rate = 10<sup>-4</sup>) as was done in the original paper. While the model in the paper had an accuracy of 48.2%, the model in this repository only achieves 46.9%. For reasons this could be happening, refer to the notes section for the six pixel judge model above.
