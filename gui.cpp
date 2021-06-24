// GUI for blob detection

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/tracking.hpp>
#include <math.h> 

using namespace cv;

cv::Mat frame;
cv::Mat img;
cv::Mat imageROI;
Mat erosion_dst, dilation_dst;
int applyCLAHE = 0;
int threshMin = 0;
int applyotsu = 0;
int roi_top = 1560;
int roi_bottom = 1560;
int roi_left = 2104;
int roi_right = 2104;
int runbutton = 0;
int boxselect = 0;
int erosion_elem = 0;
int erosion_size = 0;
int dilation_elem = 0;
int dilation_size = 0;
int opening_elem = 0;
int closing_elem = 0;
int opening_size = 0;
int closing_size = 0;
int detect_interval = 5;
int const max_elem = 2;
int const max_kernel_size = 21;


// Function that the window "Blob Detection" uses

void Control( int, void* ) {

	int erosion_type = 0;
	int dilation_type = 0;

	
	//CLAHE (Contrast Limited Adaptive Histogram Equalization)
	
	if (applyCLAHE) {
	
		Ptr<CLAHE> clahe = createCLAHE();
		
		clahe->setClipLimit(5);
		
		clahe->apply(frame, frame);
	}


	// Binary Threshold


	if (threshMin > 0) {
		threshold(frame, frame, threshMin, 255, THRESH_BINARY);
	}


	// Otsu Threshold

	if (applyotsu) {
	
		threshold(frame, frame, 0, 255, THRESH_BINARY | THRESH_OTSU);
	}

	
	// Morphological Erosion
	
	if (erosion_elem == 0) { erosion_type = MORPH_RECT; }
	else if (erosion_elem == 1) { erosion_type = MORPH_CROSS; }
	else if (erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }
	
	Mat element1 = getStructuringElement( erosion_type,
		       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
		       Point( erosion_size, erosion_size ) );
		       
	erode(frame, frame, element1);

	
	// Morphological Dilation	
	
	if (dilation_elem == 0) { dilation_type = MORPH_RECT; }
	else if (dilation_elem == 1) { dilation_type = MORPH_CROSS; }
	else if (dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }
	
	Mat element2 = getStructuringElement( dilation_type,
	       Size( 2*dilation_size + 1, 2*dilation_size+1 ),
	       Point( dilation_size, dilation_size ) );
	       
	dilate(frame, frame, element2);

	
	// Morphological Opening
	
	Mat element3 = getStructuringElement( opening_elem, Size( 2*opening_size + 1, 2*opening_size+1 ), Point( opening_size, opening_size ) );
	
	morphologyEx(frame, frame, MORPH_OPEN, element3);
	
	
	// Morphological Closing
	
	Mat element4 = getStructuringElement( closing_elem, Size( 2*closing_size + 1, 2*closing_size+1 ), Point( closing_size, closing_size ) );
	
	morphologyEx(frame, frame, MORPH_CLOSE, element4);
	
	
	// If the Run trackbar is 1, close window and start blob detection
	
	if (runbutton == 1)
		cv::destroyAllWindows ();
	
	//imshow("Blob Detection", frame);
	imshow("Preview", frame);
	imshow("Original", img);
	
}

int main() {

	cv::VideoCapture input("NIR_1.mp4");
		
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	
	// Init the OpenCV window with the trackbars and resize it
	
	namedWindow("Blob Detection", WINDOW_NORMAL);
	
	int desiredWidth=640, desiredheight=480;
	
	resizeWindow("Blob Detection", desiredWidth, desiredheight);
	
	// Init the window that shows the preview video which allows to check how different operations influence the preprocessed frames
	
	namedWindow("Preview", WINDOW_NORMAL);
	resizeWindow("Preview", desiredWidth, desiredheight);
	
	
	// Init the window that show the original video for comparison with the preview
	
	namedWindow("Original", WINDOW_NORMAL);
	resizeWindow("Original", desiredWidth, desiredheight);
	
	
	// Create the trackbars to control the values of the operations
	
	createTrackbar( "CLAHE", "Blob Detection", &applyCLAHE, 1, Control);
	
	createTrackbar( "Binary Threshold", "Blob Detection", &threshMin, 255, Control);

	createTrackbar( "Otsu Threshold", "Blob Detection", &applyotsu, 1, Control);
	
	createTrackbar( "Erode kernel type:\n 0: Rect \n 1: Cross \n 2: Ellipse", "Blob Detection",
	&erosion_elem, max_elem,
	Control );
	
	createTrackbar( "Erode kernel size:\n 2n +1", "Blob Detection",
	&erosion_size, max_kernel_size,
	Control );
	
	createTrackbar( "Dilation kernel type:\n 0: Rect \n 1: Cross \n 2: Ellipse", "Blob Detection",
	&dilation_elem, max_elem,
	Control );
	
	createTrackbar( "Dilation kernel size:\n 2n +1", "Blob Detection",
	&dilation_size, max_kernel_size,
	Control );
	
	createTrackbar( "Opening kernel type:\n 0: Rect \n 1: Cross \n 2: Ellipse", "Blob Detection",
	&opening_elem, max_elem,
	Control );
	
	createTrackbar( "Opening kernel size:\n 2n +1", "Blob Detection",
	&opening_size, max_kernel_size,
	Control );
	
	createTrackbar( "Closing kernel type:\n 0: Rect \n 1: Cross \n 2: Ellipse", "Blob Detection",
	&closing_elem, max_elem,
	Control );
	
	createTrackbar( "Closing kernel size:\n 2n +1", "Blob Detection",
	&closing_size, max_kernel_size,
	Control );
	
	createTrackbar( "Box on Blob?", "Blob Detection", &boxselect, 1, Control);
	
	createTrackbar( "Run?", "Blob Detection", &runbutton, 1, Control);
	
	
	// Loop for specifying the parameters on the GUI
	
	for (;;) {
	
		// Read each frame
	
		if (!input.read(img))
			break;
		
		cv::cvtColor(img, img, COLOR_BGR2GRAY);
		
		imageROI = img.clone();
		
		// Crop the image
		
		Rect Rec(430, 530, 1280, 800);     //x,y,width,height
		rectangle(imageROI, Rec, Scalar(255), 1, 8, 0);

		frame = imageROI(Rec);
				
		// Call the Control function 
		
	   	Control(0,0);
	   	
	   	if (runbutton) 
	   		break;
		
		//char c = cv::waitKey(0);
		
		//if (c = 27)  //27 is ESC code
		//	break;

		int keyboard = waitKey(30);
        if (keyboard == 'q' || keyboard == 27)
            break;
			
	}
	

	cv::VideoCapture video("NIR_1.mp4");
	
	cv::destroyAllWindows();
	
	namedWindow("Contours", WINDOW_NORMAL);
	resizeWindow("Contours", desiredWidth, desiredheight);
	
	int erosion_type = 0;
	int dilation_type = 0;
	

	// Print values used
	
	printf("CLAHE: %d \n", applyCLAHE);
	printf("Threshold: %d \n", threshMin);
	printf ("Otsu: %d \n",applyotsu);
	printf("Erosion type & size: %d %d \n", erosion_elem, erosion_size);
	printf("Dilation typ e& size: %d %d \n", dilation_elem, dilation_size);
	printf("Opening type & size: %d %d \n", opening_elem, opening_size);
	printf("Closing type & size: %d %d \n", closing_elem, closing_size);


    // Apply everything on video
	
	for (;;) {
	
		if (!video.read(img))
			break;
		
		
		// Crop the image
		
		Rect Rec(430, 530, 1280, 800);     //x,y,width,height
		rectangle(img, Rec, Scalar(255), 1, 8, 0);

		imageROI = img(Rec);


		frame = imageROI.clone();

		cv::cvtColor(frame, frame, COLOR_BGR2GRAY);


		// Perform the selected operations
		
		if (applyCLAHE) {

			Ptr<CLAHE> clahe = createCLAHE();

			clahe->setClipLimit(5);

			clahe->apply(frame, frame);
		}
	   	
		// Binary Threshold

		if (threshMin > 0) {
			threshold(frame, frame, threshMin, 255, THRESH_BINARY);
		}


		// Otsu Threshold


		if (applyotsu) {

			threshold(frame, frame, 0, 255, THRESH_BINARY | THRESH_OTSU);
		}


		// Morphological Erosion

		if (erosion_elem == 0) { erosion_type = MORPH_RECT; }
		else if (erosion_elem == 1) { erosion_type = MORPH_CROSS; }
		else if (erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }

		Mat element1 = getStructuringElement( erosion_type,
		       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
		       Point( erosion_size, erosion_size ) );

		erode( frame, frame, element1 );


		// Morphological Dilation	

		if (dilation_elem == 0) { dilation_type = MORPH_RECT; }
		else if (dilation_elem == 1) { dilation_type = MORPH_CROSS; }
		else if (dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }

		Mat element2 = getStructuringElement( dilation_type,
		Size( 2*dilation_size + 1, 2*dilation_size+1 ),
		Point( dilation_size, dilation_size ) );

		dilate( frame, frame, element2 );

		
		// Morphological Opening

		Mat element3 = getStructuringElement( opening_elem, Size( 2*opening_size + 1, 2*opening_size+1 ), Point( opening_size, opening_size ) );

		morphologyEx(frame, frame, MORPH_OPEN, element3);


		// Morphological Closing

		Mat element4 = getStructuringElement( closing_elem, Size( 2*closing_size + 1, 2*closing_size+1 ), Point( closing_size, closing_size ) );

		morphologyEx(frame, frame, MORPH_CLOSE, element4);


		// Find the contours

	   	findContours(frame, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);


		//cv::cvtColor(imageROI, imageROI, COLOR_GRAY2BGR);
		
		
		Rect box;
		

		// Compute the centers of the contours/blobs
		
		std::vector<cv::Moments> mu(contours.size());
		
		for (int i = 0; i<contours.size(); i++ ) { 

			mu[i] = moments( contours[i], false ); 
		}

		
		std::vector<cv::Point2f> mc(contours.size());
		
		for (int i = 0; i<contours.size(); i++) { 

			mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 );
		}


		// Draw contours 
		
		for (int i = 0; i < contours.size(); i++) {
		
			if (boxselect) {

				box = boundingRect(contours[i]); 
				rectangle(imageROI, box, cv::Scalar(255,0,0), 1, 8, 0);
			}

			
			drawContours(imageROI, contours, i, cv::Scalar(0,0,0), 1, 8, hierarchy, 0);
			circle( imageROI, mc[i], 2, cv::Scalar(255,0,0), -1, 8, 0 );
		}

		imshow("Contours", imageROI);

		int keyboard = waitKey(30);
        if (keyboard == 'q' || keyboard == 27)
            break;

	}
	
	 
} 


