Q: DiffSo network may not be suitable for resource-constrained environments.
A: We modified the framework diagram and added the main work in different power consumption environments. In the IoT environment, the main work is data preprocessing and encryption. Due to the limited computing power of IoT devices, we propose a lightweight GMEVCS for fast encryption of sensitive data. In the edge-cloud environment, the main work is decryption and image recovery. Due to the high computing power, we propose a DiffSo model to restore the original image.
Therefore, the lightweight of this paper is mainly reflected in the encryption and decryption process in the IoT environment. The DiffSo model runs in an edge-cloud environment with high computing power.

Q: The paper mentions generating two shares as the output of the encryption step, but this choice seems arbitrary and is not well-justified.
A: Although there are different choices of the number of share images, more share images will increase the storage and transmission overhead, which is not friendly to lightweight devices.

Q: When you remap the RGB to HSV and especially the brightness values from 0-255 to 0-63, then isn’t there a loss of accuracy or an increased chance of errors?
A: Due to EVCS encryption (Fig. 3), it is necessary to compress the pixel values to 1/4 of the original, which will lose information. It is the process of remapping to HSV that allows the decrypted image to retain more information (Table VI).

Q: Does this have an impact on the robustness of security guarantees if preprocessing steps are tampered with and/or bypassed in real-world settings?
A: We added a security analysis. The preprocessing process does not involve security, and security is guaranteed by GMEVCS encryption. Less than k share images cannot recover the original image, and modifying with any share image will cause decryption to fail.


Q: Have you conducted performance evaluations of the framework, specifically the DiffSo model, on IoT devices such as Raspberry Pi or NVIDIA Jetson Nano? If not, how would you address potential limitations in inference time, memory usage, and energy consumption?
A: We do not have these devices. The encryption of VCS involves only simple image processing operations, not complex mathematical operations. Decryption only requires superimposing a sufficient number of share images (boolean OR operation). Since the encryption and decryption processes are simple, they will not put pressure on low-power devices. At the same time, the halftone process also compresses the size of the image, reducing memory usage. 

Q: The method does not appear to provide a significant improvement over MSPRL. 
A: We further improved the DiffSo model, proposed to preprocess the decrypted image by Gaussian blur, convert the halftone image to a continuous-tone image, and then recover the decrypted image as shown in Fig. 2. The experimental results (Tabel II and Tabel III) show that our DiffSo achieves better Image Quality Assessment performance.

Q: How accurately can decrypted images be recognized by a classifier trained on the original images?
A: We conducted face recognition experiments. We used the pre-trained FaceNet model to perform face recognition on the LFW dataset. Table IV shows the face recognition accuracy of the unrecovered image, the MSPRL restored image, and the DiffSo restored image. The experimental results show that the recognition accuracy of MSPRL will lose 2.5%. While DiffSo only loses 0.6%. This indicates that the quality of the recovered image by DiffSo is better and more suitable for subsequent tasks in the cloud.
We also compared the recognition performance of original and restored images using the Diabetic Retinopathy Diagnosis Dataset on the ResNet18 model. Tabel V demonstrates that our method does not affect the disease diagnosis rate.


Q: Can you compare against [35] to justify the proposed DiffSo method?
A: We compared the performance of DiffSo and [35] in Tabel II. The experimental results show that [35] lacks the correct guidance information, resulting in lower image quality.

Q: The addition of a diffusion model may introduce new vulnerabilities, such as the potential for adversarial attacks or information leakage during reconstruction.
A: The deep learning model processes the decrypted image and is only used to restore the original image. In Fig. 12, we show the robustness of the DiffSo model by flip, jpeg and noise attacks. The results show that our model is robust.
Due to space limitations, we did not discuss the security of the deep learning model. We will conduct research on the robustness of adversarial attacks in future work.

