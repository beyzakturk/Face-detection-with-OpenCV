# OpenCV Face Recognition 

In today’s tutorial, you will learn how to perform face recognition using the OpenCV library.

Well, keep in mind that the dlib face recognition post relied on two important external libraries:

1. dlib (obviously)
2. face_recognition (which is an easy to use set of face recognition utilities that wraps around dlib)
 
While we used OpenCV to facilitate face recognition, OpenCV itself was not responsible for identifying faces.

In today’s tutorial, we’ll learn how we can apply deep learning and OpenCV together (with no other libraries other than scikit-learn) to:
1. Detect faces
2. Compute 128-d face embeddings to quantify a face
3. Train a Support Vector Machine (SVM) on top of the embeddings
4. Recognize faces in images and video streams
All of these tasks will be accomplished with OpenCV, enabling us to obtain a “pure” OpenCV face recognition pipeline.

## How OpenCV’s face recognition works

![Image](https://pyimagesearch.com/wp-content/uploads/2018/09/opencv_face_reco_facenet.jpg)

In order to build our OpenCV face recognition pipeline, we’ll be applying deep learning in two key steps:

1. To apply face detection, which detects the presence and location of a face in an image, but does not identify it
2. To extract the 128-d feature vectors (called “embeddings”) that quantify each face in an image
I’ve discussed how OpenCV’s face detection works previously, so please refer to it if you have not detected faces before.

The model responsible for actually quantifying each face in an image is from the OpenFace project, a Python and Torch implementation of face recognition with deep learning. This implementation comes from Schroff et al.’s 2015 CVPR publication, FaceNet: A Unified Embedding for Face Recognition and Clustering.

Reviewing the entire FaceNet implementation is outside the scope of this tutorial, but the gist of the pipeline can be seen in Figure 1 above.

First, we input an image or video frame to our face recognition pipeline. Given the input image, we apply face detection to detect the location of a face in the image.

Optionally we can compute facial landmarks, enabling us to preprocess and align the face.

Face alignment, as the name suggests, is the process of (1) identifying the geometric structure of the faces and (2) attempting to obtain a canonical alignment of the face based on translation, rotation, and scale.

While optional, face alignment has been demonstrated to increase face recognition accuracy in some pipelines.

After we’ve (optionally) applied face alignment and cropping, we pass the input face through our deep neural network:

![Image](https://pyimagesearch.com/wp-content/uploads/2018/09/opencv_face_reco_training.png)

The FaceNet deep learning model computes a 128-d embedding that quantifies the face itself.

But how does the network actually compute the face embedding?

The answer lies in the training process itself, including:

1. The input data to the network
2. The triplet loss function
To train a face recognition model with deep learning, each input batch of data includes three images:

1. The anchor
2. The positive image
3. The negative image
The anchor is our current face and has identity A.

The second image is our positive image — this image also contains a face of person A.

The negative image, on the other hand, does not have the same identity, and could belong to person B, C, or even Y!

The point is that the anchor and positive image both belong to the same person/face while the negative image does not contain the same face.

The neural network computes the 128-d embeddings for each face and then tweaks the weights of the network (via the triplet loss function) such that:

1. The 128-d embeddings of the anchor and positive image lie closer together
2. While at the same time, pushing the embeddings for the negative image father away
In this manner, the network is able to learn to quantify faces and return highly robust and discriminating embeddings suitable for face recognition.

And furthermore, we can actually reuse the OpenFace model for our own applications without having to explicitly train it!

Even though the deep learning model we’re using today has (very likely) never seen the faces we’re about to pass through it, the model will still be able to compute embeddings for each face — ideally, these face embeddings will be sufficiently different such that we can train a “standard” machine learning classifier (SVM, SGD classifier, Random Forest, etc.) on top of the face embeddings, and therefore obtain our OpenCV face recognition pipeline.

If you are interested in learning more about the details surrounding triplet loss and how it can be used to train a face embedding model, be sure to refer to my previous blog post as well as the Schroff et al. publication.

