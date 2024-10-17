
# Speech Enhancement Using Deep Learning - Denoising Project

This was the final project in my BMME 575 course, Practical Machine Learning for Biosignal Analysis, at The University of North Carolina at Chapel Hill, taught by Dr. Arian Azarang. The primary objective was to implement and evaluate a deep learning model for speech enhancement, specifically to denoise audio signals. 

## Project Description

Our team aimed to implement the speech denoising architecture described in the IEEE paper *Noisy speech enhancement based on correlation canceling/log-MMSE hybrid method* (Asbai et al., 2023). The project leveraged an open-source GitHub repository that had implemented a similar model. However, a significant amount of our time was dedicated to understanding, debugging, and modifying the provided code to align with our objectives. The complexity of setting up the code and running it via shell scripts was a major challenge, requiring considerable effort in debugging, which I primarily led.

## Team Members
- **Micah Baldonado (Team Leader)** - Main contributions in code debugging, STOI calculation for LogMMSE, and implementing the speech enhancement model.
- Anushka Deshmukh - Data preprocessing. Assistance with the post-model training analysis.
- Hasan Dheyaa and William McLain - Post-model training analysis and model performance optimization.

## Additional Resources

- The project presentation can be found [here](bmme575_final_presentation.pdf).
- The final report can be found [here](575_report.pdf).

### Key Contributions:
1. **Paper Selection**: I identified the target paper for our project, which guided the model implementation.
2. **Debugging and Running the Model**: 
   - One of the most challenging tasks was getting the open-source code to run correctly. The code was outdated, and I spent several nights working through the errors in the shell scripts.
   - Once the code was running, I was responsible for making sure my team members had the resources they needed to train the model, giving my guidance and ensuring the could be used for further analysis.
3. **STOI Calculations for LogMMSE**: 
   - During the first phase of the project, I calculated the Short-Time Objective Intelligibility (STOI) metric for the speech enhancement performance of the LogMMSE method. This was crucial as it set a performance benchmark that we used when we later trained the deep learning model.
   - These metrics were calculated for noisy data at -5dB and 0dB signal-to-noise ratio (SNR) levels.
4. **Model Training**:  Once the deep learning model was running, I passed on the tasks of optimizing the model to my teammates. However, I was instrumental in getting the model to train and produce preliminary results, such as the model's loss and performance metrics. "
5. **Presentation**: As the team leader, I guided the presentation of our methodology, explaining the architecture of the model from the ground up. Because I wanted all the students in the class to have some base level understanding, I also explained how the input data eventually is processed by the model so that it can learn to enhance speech from mixed sound data (mixture between clean speech and noise).

## Model Evaluation and Results

The model was evaluated based on two objective metrics:
- **STOI (Short-Time Objective Intelligibility)**: This metric measures the intelligibility of the enhanced speech.
- **PESQ (Perceptual Evaluation of Speech Quality)**: This metric evaluates the overall speech quality.

### Results:
- The deep learning model, while showing promise, had limitations in performance when compared to the LogMMSE method. The results demonstrated that, particularly at the -5dB SNR level, the LogMMSE method performed better in terms of STOI, while the DNN model had higher PESQ values at 0dB.
- One of the main challenges was that our model had significantly less training time (approximately 3 hours) compared to the 650 hours of training described in the original paper, which impacted the model’s performance.

## Limitations:
- **Training Time**: The model was trained for significantly less time than the original paper suggested, which likely affected the overall performance.
- **Over-Smoothing**: The DNN model tended to over-smooth the enhanced speech, which led to reduced intelligibility in some cases.
- **Data Complexity**: The dataset used for training was much smaller and less varied than that in the original paper, which limited the model's ability to generalize.

## Conclusion
This project provided hands-on experience with deep learning techniques for speech enhancement. Despite the challenges we faced, the project was successful in implementing and evaluating a deep learning-based speech enhancement model. Our analysis revealed key areas for future improvement, including increasing training time and expanding the dataset.

The experience allowed me to further develop my machine learning skills, particularly in debugging and working with shell scripts, while also gaining a deeper understanding of speech enhancement techniques and their practical applications.

## References
Asbai N, Zitouni S, Bounazou H, Yahi A. Noisy speech enhancement based on correlation canceling/log-MMSE hybrid method. Multimed Tools Appl 2023;82:5803–21. https://doi.org/10.1007/s11042-022-13591-8.
