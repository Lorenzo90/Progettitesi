package com.example.lorenzoorlandi.opencvproj;

import android.hardware.Camera;
import android.nfc.Tag;
import android.os.Environment;
import android.support.annotation.NonNull;
import android.support.v7.app.ActionBarActivity;
import android.os.Bundle;
import android.view.Menu;
import android.view.MenuItem;
import org.opencv.calib3d.Calib3d;


import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.ViewGroup;
import android.view.WindowManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.DMatch;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.KeyPoint;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;

public class MainActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "OpenCVlol";
    private Mat mRgba;
    private Mat mIntermediateMat;
    private Mat mGray;
    List<Mat> descriptorsList;
    private CameraBridgeViewBase mOpenCvCameraView;
    private VideoCapture mCamera;
    FeatureDetector detector;
    DescriptorExtractor extractor;
    DescriptorMatcher matcher;
    MatOfKeyPoint keypoints1;
    MatOfKeyPoint keypoints2;
    LinkedList<DMatch> listOfGoodMatches = new LinkedList<>();
    Mat descriptortwo;
    Mat descriptorone;
    Mat[] descriptors;
    Mat Hsvimage;
    Mat mRgb2;
    Mat m;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
       // mCamera = new VideoCapture();
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.hello_open_cv_layout);
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.HelloOpenCvView);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
       // mOpenCvCameraView.set(Highgui.CV_CAP_PROP_FRAME_WIDTH, 1000);
        //mOpenCvCameraView.set(Highgui.CV_CAP_PROP_FRAME_HEIGHT, 800);
       // mOpenCvCameraView.setMaxFrameSize(500,500);
    }



    @Override
    public void onResume() {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_10, this, mLoaderCallback);
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        //Log.d("CIAo", "fregati");

        //Dichiaro il detector delle feature i descrittori e infine il matcher del descrittore
        detector = FeatureDetector.create(FeatureDetector.ORB);
        extractor = DescriptorExtractor.create(DescriptorExtractor.BRIEF);
        matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMINGLUT);

        //assegno le matrici con i vari valori del frame della camera  in questo caso utilizzo 3 canali RGB e un canale gray
        mRgba = new Mat(height, width, CvType.CV_8UC3);
        mGray = new Mat(height, width, CvType.CV_8UC1);

       /* //Stringa del file da caricare, in questo caso l'immagine campione utilizzata per trovare i Keypoints
        String templateFile = Environment.getExternalStorageDirectory().getAbsolutePath()+"/Template/"+"imm.png";
        m= new Mat();
        m=Highgui.imread(templateFile, Highgui.CV_LOAD_IMAGE_COLOR);
        Log.d(TAG,"Grandezza immagine:" + templateFile + Integer.toString(m.height())+"*"+Integer.toString(m.width()));

        //Gaussian Blur
        Mat dst = new Mat();
        Imgproc.GaussianBlur( m, dst, new  Size(5,5), 2, 2);

        //Assegno le varie matrici dei descrittori e dei punti chiave  gli estraggo solo nel caso della prima immagine
        descriptorone = new Mat();
        keypoints1 = new MatOfKeyPoint();
        keypoints2 = new MatOfKeyPoint();
        detector.detect(m, keypoints1);
        extractor.compute(m, keypoints1, descriptorone);
        KeyPoint tkp1[] = keypoints1.toArray();
        Log.d(TAG, "NUMERO KEYPOINT ESTRATTI dalla prima picture:" + Integer.toString(tkp1.length) );
*/
    }

    public void onCameraViewStopped() {
        mRgba.release();
        mGray.release();
    }


    //Funzione che riceve in ingresso il frame dalla camera per poi poterlo processare
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        Mat mRgbcann = new Mat();

        // All'interno di questa funzione processo i vari frame che arrivano dalla fotocamera
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

        //Funzione Segmentazione dei colori
        Mat mRgba1 = FunctionSegmentazione(mRgba);

        // Mat mRgba1 = Proccanny(inputFrame.gray());

    //    Mat mRgba1 = Extractdesc(mRgba);

       // m = new Mat();
        //  String templateFile = Environment.getExternalStorageDirectory().getAbsolutePath()+"/Template/"+"imm.png";
        //   m =Highgui.imread(templateFile, Highgui.CV_LOAD_IMAGE_COLOR);
        //   String templateFilew = Environment.getExternalStorageDirectory().getAbsolutePath() + "/Template/" + "imm9.png";
       /* Mat mq = Highgui.imread(templateFilew, Highgui.CV_LOAD_IMAGE_COLOR);*/

       // Mat mRgba1 = check_matching_template(m ,mRgba);
        // Highgui.imwrite(templateFilew, mRgba );

        //  Mat mRgba1 = calHist(mRgba);


        return mRgba1;// Rect_recognize(mRgba1);
    }

    //Template Matching funzione
    private Mat check_matching_template(Mat m, Mat mRg5) {
        //Metodo per fare il matching del template
        int match_method = Imgproc.TM_SQDIFF;

        int larghezza = 20;
        int lunghezza = 80;

        Mat graymask = new Mat();
        Imgproc.cvtColor(m ,graymask , Imgproc.COLOR_RGB2GRAY);
        Mat mask = new Mat();
        //Imgproc.cvtColor(mRgba ,mask , Imgproc.COLOR_RGB2BGR);
        //Mat dst = new Mat(larghezza, lunghezza,CvType.CV_8UC1);
        //Imgproc.resize(graymask ,dst, new Size(larghezza, lunghezza));



        // Create the result matrix
        Mat mRg4 = new Mat();
        mRg4 = mRg5;
        Imgproc.cvtColor(mRg5,mRg4, Imgproc.COLOR_RGB2GRAY);
        int result_cols = mRg4.cols() - graymask.cols() + 1;
        int result_rows = mRg4.rows() - graymask.rows() + 1;
        Mat result = new Mat(result_rows, result_cols, CvType.CV_32F);

        // Do the Matching and Normalize
        Imgproc.matchTemplate(mRg4, graymask, result, match_method);
        //Core.normalize(result, result, 0, 1, Core.NORM_MINMAX, -1, new Mat());

        // Localizing the best match with minMaxLoc
        Core.MinMaxLocResult mmr = Core.minMaxLoc(result);

        Point matchLoc;
        if (match_method == Imgproc.TM_SQDIFF || match_method == Imgproc.TM_SQDIFF_NORMED) {
            matchLoc = mmr.minLoc;
            Log.d(TAG+"Val", "Valore min " + Double.toString(mmr.minVal));
            Log.d(TAG+"Val", "Valore max " + Double.toString(mmr.maxVal));
        } else {
            matchLoc = mmr.maxLoc;
        }

        String outFile = Environment.getExternalStorageDirectory().getAbsolutePath()+"/Template/"+"imm6.png";
        Highgui.imwrite(outFile,mRg5);
       // Rect roi = new Rect((int) matchLoc.x, (int) matchLoc.y, dst.cols(), dst.rows());
        Log.d(TAG+"Mat", "Valore larghezza roi:" + Integer.toString(graymask.cols()));
        Log.d(TAG + "Mat", "Valore lunghezza roi:" + Integer.toString(graymask.rows()));
        Log.d(TAG + "Mat", "roix:" + Integer.toString((int) matchLoc.x));
        Log.d(TAG+"Mat", "roiy:" + Integer.toString((int) matchLoc.y));
        if(mmr.minVal < 1500000) {
            Core.circle(mRg5, new Point((int) matchLoc.x + (int) (graymask.cols() / 2), (int) matchLoc.y + (int) (graymask.rows() / 2)), 10, new Scalar(255, 0, 0, 255), 2);
        }
        return mRg5;
    }

    //Funzione Utilizza Canny per recuperare i contorni dell'immagine
    private Mat Proccanny(Mat gray){
        //Variabili d'appoggio
        Mat Canny = new Mat();

        // Applico Canny alla matrice gray
        Imgproc.Canny( gray,  Canny, 300, 600, 5, true);

        // Assegno le Matrici che utilizzo nella funzione come variabili appoggio
        Hsvimage = new Mat();
        Mat dst = new Mat();
        Mat filteyell;
        Mat filtered;
        Mat filteblank;
        Mat filtertot;
        Mat elementerode;
        Mat element;
        List<MatOfPoint> contour = new ArrayList<MatOfPoint>();
        //Eseguo il Blur dell'immagine
        Imgproc.GaussianBlur(mRgba, dst, new Size(5, 5), 2, 2);

        // converto il frame in ingresso in HSV per la segmentazione
        Imgproc.cvtColor(mRgba, Hsvimage, Imgproc.COLOR_RGB2HSV_FULL, 3);

        //Assegno la matrice filtered che diventera la mia maschera
        filtered = new Mat(Hsvimage.height(), Hsvimage.width(),  CvType.CV_8UC1);
        filteyell = new Mat(Hsvimage.height(), Hsvimage.width(),  CvType.CV_8UC1);
        filtertot = new Mat(Hsvimage.height(), Hsvimage.width(),  CvType.CV_8UC1);

        //Estraggo le componenti per me informative in questo caso il colore giallo  tramite la prima componente HUE
        Core.inRange(Hsvimage, new Scalar(34, 60, 0), new Scalar(39, 255, 255), filteyell);
        Core.inRange(Hsvimage, new Scalar(250, 70, 0), new Scalar(254, 255, 255), filtered);
        //Core.inRange(Hsvimage, new Scalar(240, 0, 0), new Scalar(255, 50, 255), filteblank);

        //Faccio una dilatazione con un elemento rettangolare per considerare solo le parti intorno al colore
        //selezionato
        element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(9, 9));

        //Faccio l'unione delle due maschere
        Core.bitwise_or(filtered, filteyell, filtertot);
       Imgproc.erode(filtertot, filtertot, element);

        Mat Canny3 = new Mat();
        Core.bitwise_and(Canny, filtertot, Canny3);

        Mat Canny2 = new Mat();
        Canny.copyTo(Canny2, Canny3);


        Imgproc.GaussianBlur(gray, dst, new Size(5, 5), 2, 2);

        final Mat hierarchy = new Mat();
        Imgproc.findContours(Canny3 ,contour ,hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE );




        return Canny2;
    }

    //Funzione che esegue la segmentazione dei colori nell'immagine
    private Mat FunctionSegmentazione(Mat mRgba) {

        // Assegno le Matrici che utilizzo nella funzione come variabili appoggio
        Hsvimage = new Mat();
        Mat dst = new Mat();
        Mat filteyell;
        Mat filtered;
        Mat filtered1;
        Mat filtered2;
        Mat filtergrey;
        Mat filtertot;
        Mat elementerode;
        Mat element;

        //Eseguo il Blur dell'immagine
        //Imgproc.GaussianBlur(mRgba, dst, new Size(7, 7), 5, 2);

        // converto il frame in ingresso in HSV per la segmentazione
        Imgproc.cvtColor(mRgba, Hsvimage, Imgproc.COLOR_RGB2HSV_FULL, 3);

        //Assegno la matrice filtered che diventera la mia maschera
        filtered = new Mat(Hsvimage.height(), Hsvimage.width(),  CvType.CV_8UC1);
        filtered1 = new Mat(Hsvimage.height(), Hsvimage.width(),  CvType.CV_8UC1);
        filtered2 = new Mat(Hsvimage.height(), Hsvimage.width(),  CvType.CV_8UC1);
        filteyell = new Mat(Hsvimage.height(), Hsvimage.width(),  CvType.CV_8UC1);
        filtertot = new Mat(Hsvimage.height(), Hsvimage.width(),  CvType.CV_8UC1);
        filtergrey = new Mat(Hsvimage.height(), Hsvimage.width(),  CvType.CV_8UC1);










        //Estraggo le componenti per me informative in questo caso il colore giallo  tramite la prima componente HUE
        //PARAMETRI FONDAMENTALI DEL SISTEMA PER FILTRARE I COLORI DA NOI SCELTI
        Core.inRange(Hsvimage, new Scalar(34, 60, 0), new Scalar(65, 255, 255), filteyell);
        Core.inRange(Hsvimage, new Scalar(250, 70, 52), new Scalar(255, 255, 255), filtered1);
        Core.inRange(Hsvimage, new Scalar(0, 70, 52), new Scalar(5, 255, 255), filtered2);
        Core.inRange(Hsvimage, new Scalar(20, 5, 50), new Scalar(330, 70, 250), filtergrey);
        Imgproc.dilate(filtergrey, filtergrey, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5)));



        Core.bitwise_or(filtered1, filtered2, filtered);

        Imgproc.dilate(filtergrey, filtergrey, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(7, 7)));

        Core.bitwise_and(filtergrey, filtered, filtered);

        //Creo un elemento a forma ellittica e faccio l'erosione per togliere falsi positivi
        elementerode = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new  Size(4, 3));
        //Imgproc.erode(filteyell, filteyell, elementerode);
        //Imgproc.dilate(filtered, filtered, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new  Size(7, 7)));

        //Faccio una dilatazione con un elemento rettangolare per considerare solo le parti intorno al colore
        //selezionato
        element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        Imgproc.erode(filteyell, filteyell, element);
        //Imgproc.dilate(filtered, filtered, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(4, 8)));
        Imgproc.dilate(filteyell, filteyell, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(10, 5)));

        //Faccio l'unione delle due maschere
       // Core.bitwise_or(filtered, filteyell, filtertot);
        elementerode = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new  Size(3, 3));
        //Imgproc.erode(filtertot, filtertot, elementerode);



        //Applico il filtro all'immagine in modo tale da estropolare le mie compontenti di interessi (giallo)
        mRgb2 = new Mat();
        mRgba.copyTo(mRgb2,filtered);


        //disegno un rettangolo all'interno del bidone
       List<Point> contours = Extract_Area(filteyell, filtered, Hsvimage);
        if(contours != null) {
            Log.d(TAG + "str", "numero punti per 2 +1" + Integer.toString(contours.size()));
            for (int i = 0; i < contours.size(); i = i + 2) {
                try {
                    Core.circle(mRgb2, contours.get(i), 1, new Scalar(0, 255, 255), 1);
                } catch (Exception e) {
                    Log.d(TAG, e.toString());
                }
            }
            //Funzione per Estrarre le feature e fare e fare il matching
            // mRgb2 = Extractdesc(mRgb2, filtyell);
        }
        return mRgb2 ;
    }

    // Funzione che estrae le aree di interesse gialle e rosse ne verifica la posizione e
    //restituisce i punti dove sta il bidone utilizzati per disegnare il rettangolo
    List<Point> Extract_Area( Mat imageyell, Mat imagered, Mat Hsv2){

        //Variabili d'appoggio
        Mat edges = new Mat();
        List<MatOfPoint> contours= new ArrayList<MatOfPoint>();
        List<MatOfPoint> contours1 =  new ArrayList<MatOfPoint>();
        List<MatOfPoint> maxContour = new ArrayList<MatOfPoint>();
        List<Integer> point = new ArrayList<Integer>();
        double maxContourArea = 0;
        double maxAreaIdx = 0;
        MatOfPoint2f approxCurve;
        Mat mask;
        Mat contourRegion;
        Point[] points_contour;
        int[] punticontornogial = null;
        int[] punticontornoross;
        Boolean stato ;
        Rect roi;
        List<Point> rect= new ArrayList<>();
        double[] color;
        Point lol =null;
        Mat cropped = null;


        //Estraggo i contorni dalla mia maschera
        Log.d(TAG, "trovo contorno");
        Imgproc.findContours(imageyell, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
        Imgproc.findContours(imagered, contours1, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
        //Ciclo sulle immagini gialle
        for (int idx = 0; idx < contours.size(); idx++) {
            Log.d(TAG, "entrato primi cont");
            MatOfPoint contour = contours.get(idx);

            //funzione per estrarre l'area
            double contourarea = Imgproc.contourArea(contour);
            int l=0;
            int support_vary = 5000;
            int yotti =0;


            //PARAMETRO FONDAMENTALE DEL SISTEMA PER FILTRARE AREE ESTRATTE TROPPO PICCOLE GIALLE
            if (contourarea > 120) {
                for (int ydx = 0; ydx < contours1.size(); ydx++) {
                    MatOfPoint contour1 = contours1.get(ydx);
                    Point[] pArray1 = contour1.toArray();
                    double contourarea1 = Imgproc.contourArea(contour1);
                    Point[] pArray = contour.toArray();
                    punticontornogial = Extractestremi(pArray);
                    yotti = ((int)((punticontornogial[3] - punticontornogial[2])/2)+punticontornogial[2]);
                    punticontornoross = Extractestremi(pArray1);
                   // Log.d(TAG, "Aree rosse:" + Double.toString(contourarea1));

                    //controllo se gli estremi delle aree rosse sono contenuti negli estremi dell'area gialla

                    stato = Checkestremi(punticontornogial,punticontornoross);
                    //PARAMETRO FONDAMENTALE DEL SISTEMA PER FILTRARE AREE ROSSE NON COMPRESE TRA QUELLA GIALLA E UN
                    // PARAMETRO DI SOGLIA
                    if(l == 2){
                        cropped= new Mat();
                        int puntox = punticontornogial[1];
                        Log.d(TAG, "puntoxmaxroi:" + Integer.toString(puntox));
                        int puntoy = punticontornogial[3];
                        Log.d(TAG, "puntoymaxroi:" + Integer.toString(puntoy));
                        int puntoxm = punticontornogial[0];
                        Log.d(TAG, "puntoxminroi:" + Integer.toString(puntoxm));
                        int puntoym = punticontornogial[2];
                        Log.d(TAG, "puntoyminroi:" + Integer.toString(puntoym));
                        Log.d(TAG, "diff:" + Integer.toString(punticontornogial[4]));
                        Hsv2.submat(punticontornogial[2],240,punticontornogial[0]+punticontornogial[4],punticontornogial[1]-punticontornogial[4]).copyTo(cropped);
                        Log.d(TAG, "Lunghezza matrice:"+ Integer.toString(cropped.rows())+"Larghezzaa matrice:"+ Integer.toString(cropped.cols()));
                        break;
                    }

                    if (((contourarea1 < contourarea) && (contourarea1> 15)) && stato && (l < 2)) {
                        // Log.d(TAG, "CONTORNI TROVATI: " + Double.toString(maxContourArea));
                        l++;
                        int rangex = punticontornoross[1] - punticontornoross[0];
                        //Log.d(TAG, "rangex:" + Integer.toString(rangex));
                        int rangey = punticontornoross[3] - punticontornoross[2];
                        //Log.d(TAG, "rangey:" + Integer.toString(rangey));
                        int puntox = punticontornoross[1];
                        Log.d(TAG, "puntoxmax:" + Integer.toString(puntox));
                        int puntoy = punticontornoross[3];
                        Log.d(TAG, "puntoymax:" + Integer.toString(puntoy));
                        int puntoxm = punticontornoross[0];
                        Log.d(TAG, "puntoxmin:" + Integer.toString(puntoxm));
                        int puntoym = punticontornoross[2];
                        Log.d(TAG, "puntoymin:" + Integer.toString(puntoym));
                        color = Hsv2.get((punticontornoross[3] - (int) rangey / 2), (punticontornoross[1] - (int) rangex / 2));

                        //Log.d(TAG+"str", "numero Hue"+Double.toString(color[0]));
                        //Log.d(TAG+"str", "numero SAt"+Double.toString(color[1]));

                        point.add(rangex);
                        point.add(rangey);
                        point.add(punticontornoross[1]);
                        point.add(punticontornoross[3]);
                        int puntocalc = Math.abs((int) (punticontornoross[3] - (int)(rangey / 2))-yotti);
                        if(puntocalc < support_vary) {
                            if(lol != null){ lol = null;}
                            lol = new Point((double) (punticontornoross[1] - rangex / 2), (double) (punticontornoross[3] - rangey / 2));
                            support_vary = puntocalc;
                        }

                        // Controllo se nei punti recuperati c e un pattern rettangolare
                    }
                }
            }
            if(lol != null) {
                rect.add(lol);
            }
        }


        if(rect.size()==0){
            return null;
        }else {
            return rect;
        }






        /*
        //Ciclo sulle immagini gialle
        for (int idx = 0; idx < contours.size(); idx++) {
            Log.d(TAG, "entrato primi cont");
            MatOfPoint contour = contours.get(idx);

            //funzione per estrarre l'area
            double contourarea = Imgproc.contourArea(contour);

            //PARAMETRO FONDAMENTALE DEL SISTEMA PER FILTRARE AREE ESTRATTE TROPPO PICCOLE GIALLE
            if (contourarea > 120) {

                //ciclo sulle aree rosse
                Log.d(TAG, "entrato secondi cont");
                for (int ydx = 0; ydx < contours1.size(); ydx++) {
                    MatOfPoint contour1= contours1.get(ydx);
                    Point[] pArray = contour.toArray();
                    Point[] pArray1 = contour1.toArray();
                    double contourarea1 = Imgproc.contourArea(contour1);
                    punticontornogial = Extractestremi(pArray);
                    punticontornoross = Extractestremi(pArray1);

                    //controllo se gli estremi delle aree rosse sono contenuti negli estremi dell'area gialla
                    stato = Checkestremi(punticontornogial,punticontornoross);

                    //PARAMETRO FONDAMENTALE DEL SISTEMA PER FILTRARE AREE ROSSE NON COMPRESE TRA QUELLA GIALLA E UN
                    // PARAMETRO DI SOGLIA
                    if (((contourarea1 < contourarea) && (contourarea1> 30)) && stato) {
                        Log.d(TAG, "CONTORNI TROVATI: " + Double.toString(maxContourArea));
                        maxContour.add(contour1);
                        int rangex = punticontornoross[1] - punticontornoross[0];
                        Log.d(TAG, "rangex:" + Integer.toString(rangex));
                        int rangey = punticontornoross[3] - punticontornoross[2];
                        Log.d(TAG, "rangey:" + Integer.toString(rangey));
                        point.add(rangex);
                        point.add(rangey);
                        point.add(punticontornoross[1]);
                        point.add(punticontornoross[3]);
                        Point lol = new Point((double)(punticontornoross[1] - (int)rangex/2),(double)(punticontornoross[3] - (int)rangey/2));
                        rect.add(lol);

                        // Controllo se nei punti recuperati c'è un pattern rettangolare
                    }
                }

            }
        }

       // Controllo di avere almeno 4 punti altrimenti il match con il bidone non
       // lo ritengo rispettato
        if(point.size()>2) {

            List<Point> rect= new ArrayList<>();

            // Controllo se nei punti recuperati c'è un pattern rettangolare
            rect =check_rect(point);

            // restituisco i punti nel caso di pattern rettangolare
            if (rect == null) {
                return null;
            }else{
                return rect;
            }
        }
       return null;*/
    }

    //Funzione per il controllo del pattern rettangolare
    private List<Point> check_rect(List<Integer> point) {
        int numberpoint = (point.size())/4;
        Point primo;
        Point secondo;
        int pointy2= 7000;
        List<Integer> pointy = new ArrayList<>();
        List<Integer> pointx = new ArrayList<>();
        List<Integer> rangex = new ArrayList<>();
        List<Integer> rangey = new ArrayList<>();
        List<Integer> indiciy = new ArrayList<>();
        List<Integer> indicix = new ArrayList<>();
        List<Point>  rect = null;


        if(numberpoint < 4){
            return null;
        }

        //Divido la lista in liste dove sono contenute le componenti
        // x e y e i vari range
        for(int y=numberpoint-1; y >= 0; y--){
            int pointy1 = point.get(y*4+3)- point.get(y*4+1)/2;
            int pointx1 = point.get(y*4+2)- point.get(y*4)/2;
            Log.d(TAG+"imp", "Puntololy:" + Integer.toString(pointy1));
            Log.d(TAG+"imp", "Puntololx:" + Integer.toString(pointx1));
           // Log.d(TAG, "Punto rangey:" + Integer.toString(point.get(y*4+1)));
           // Log.d(TAG, "Punto y:" + Integer.toString(pointy1));
            pointy.add(pointy1);
            pointx.add(pointx1);
            rangex.add(point.get(y*4)/2);
            rangey.add(point.get(y*4+1)/2);
        }

        primo = new Point(pointx.get(0),pointy.get(0));

        //Sfrutto il fatto che i punti sono ordinati rispetto a y e cerco se c'è una corrispondenza
        //quadrata nei punti
        indiciy.add(0);
        for(int i=1; i < pointy.size() ; i++){
        Boolean stato1 =((pointy.get(0) + rangey.get(0)) >= pointy.get(i)) && ((pointy.get(0) - rangey.get(0)) <=  pointy.get(i)) ;
        Boolean stato2 =((pointx.get(0) + rangex.get(0)) >= pointx.get(i)) && ((pointx.get(0) - rangex.get(0)) <=  pointx.get(i)) ;
            if(stato1){
                indiciy.add(i);
                Log.d(TAG+"imp", "inidici y:" + Integer.toString(i));
            }
            if(stato2){
                indicix.add(i);
                Log.d(TAG+"imp", "inidici x:" + Integer.toString(i));;
            }
        }

        //controllo di avere sia conponenti in x che in y
        if((indiciy.size() > 0)&&(indicix.size() > 0)) {
            //Ciclo sugli elementi ad una stessa altezza
            for (int i = 0; i < indiciy.size() ; i++){
                int indy = indiciy.get(i);
                List<Integer> x = new ArrayList<>();
                for(int y=0; y < pointx.size() ; y++) {
                    Boolean stato2 = ((pointx.get(indy) + rangex.get(indy)) >= pointx.get(y)) && ((pointx.get(indy) - rangex.get(indy)) <= pointx.get(y));
                    if (stato2 && indy != y) {
                        Log.d(TAG + "imp", "pointy sup:" + Integer.toString(pointy.get(indy)));
                        Log.d(TAG + "imp", "inidiciy inf:" + Integer.toString(pointy.get(y)));
                        Log.d(TAG + "imp", "pointx sup:" + Integer.toString(pointx.get(indy)));
                        Log.d(TAG + "imp", "inidicix inf:" + Integer.toString(pointx.get(y)));
                        Log.d(TAG + "imp", "range x:" + Integer.toString(rangex.get(indy)));
                        x.add(y);
                    }
                }
                if(x.size() > 0) {
                    List<Point> check_mask = check_str(indy, x, pointy, pointx);
                    if(check_mask != null){
                        if(rect == null){
                            rect =check_mask;
                        }else {
                            rect.addAll(check_mask);
                        }
                        Log.d(TAG+"str", "Punti assegnati");
                    }
                }
                x.clear();
            }
            return rect;
        }else{
            return null;
        }



/*
        //controllo di avere sia conponenti in x che in y e infine controllo l'elemento in diagonale
        if((indiciy.size() > 0)&&(indicix.size() > 0)){
           if(indicix.get(0) == 2){
               Boolean stato1 =((pointy.get(2) + rangey.get(2)) >= pointy.get(3)) && ((pointy.get(2) - rangey.get(2)) <=  pointy.get(3)) ;
               Boolean stato2 =((pointx.get(1) + rangex.get(1)) >= pointx.get(3)) && ((pointx.get(1) - rangex.get(1)) <=  pointx.get(3)) ;
               //Log.d(TAG, "punto diagonalex:" + Integer.toString(pointx.get(1)) +"- "+ Integer.toString(pointx.get(3)));
               //Log.d(TAG, "punto diagonaley:" + Integer.toString(pointy.get(3)) +" -"+ Integer.toString(pointy.get(2)));
               if(stato1 && stato2){
                   Log.d(TAG+"imp", "funziona");
                   secondo = new Point(pointx.get(3),pointy.get(3));
                   rect[0] =primo;
                   rect[1] =secondo;
                   return rect;
               }

           }else if(indicix.get(0) == 3){
               Boolean stato1 =((pointy.get(3) + rangey.get(3)) >= pointy.get(2)) && ((pointy.get(3) - rangey.get(3)) <=  pointy.get(2)) ;
               Boolean stato2 =((pointx.get(1) + rangex.get(1)) >= pointx.get(2)) && ((pointx.get(1) - rangex.get(1)) <=  pointx.get(2)) ;
             //  Log.d(TAG, "punto diagonalex:" + Integer.toString(pointx.get(1)) +"- "+ Integer.toString(pointx.get(2)));
              // Log.d(TAG, "punto diagonaley:" + Integer.toString(pointy.get(2)) +" -"+ Integer.toString(pointy.get(3)));
               if(stato1 && stato2){
                   Log.d(TAG+"imp", "funziona");
                   secondo = new Point(pointx.get(2),pointy.get(2));
                   rect[0] =primo;
                   rect[1] =secondo;
                   return rect;
               }


           }else{
               return null;
           }
        }else{
            return null;
        }*/

    }

    //Controllo se sono presenti striscie rosso grigrio rosso sul bidone
    private List<Point> check_str(int indy, List<Integer> indicix, List<Integer> pointy, List<Integer> pointx) {

        // Punto di riferimento
        int xiniziale = pointx.get(indy);
        int yiniziale = pointy.get(indy);
        double[] color;
        List<Point> punti = new ArrayList<>();
        int nummin = 7000;

        for(int i=0; i < indicix.size(); i++ ) {
            int xsecondo = pointx.get(indicix.get(i));
            int ysecondo = pointy.get(indicix.get(i));
            int numerogrigi=0;
            int numtot=0;
            for (int ind = yiniziale; ind < ysecondo; ind++){
                color = Hsvimage.get(ind,xiniziale);
                if ( 0 < color[1] && color[1] <= 70){// &&  (250< color[0] || color[0] <= 200 ))
                    Log.d(TAG+"str", "numero Hue"+Double.toString(color[0]));
                    Log.d(TAG+"str", "numero SAt"+Double.toString(color[1]));
                    numerogrigi++;
                }
            }
            numtot =   ysecondo -yiniziale;
            Log.d(TAG+"str", "numero grigi"+Integer.toString(numerogrigi));
            Log.d(TAG+"str", "numero tot"+Integer.toString(numtot));
            Log.d(TAG+"str", "numero salv"+Integer.toString(nummin));
            //PARAMETRO FONDAMENTALE DEL SISTEMA PER RICONOSCERE I GRIGI
            Boolean check = (int)(numtot/5) < numerogrigi;
            if((numtot < nummin)&&(check)&&(numtot>0)){
                if(punti.size() != 0){
                    punti.clear();
                }
                Log.d(TAG+"str", "numero tot ENTRATOOOOOOOOOO"+Integer.toString(numtot));
                nummin = numtot;
                punti.add(new Point((double)xsecondo,(double)ysecondo));
                punti.add( new Point((double)xiniziale,(double)yiniziale));
            }
        }
        if(punti.size() != 0){
            return punti;
        }else{
            return null;
        }

    }

    //Faccio un controllo sui colori
    //controllo se gli estremi gialli comprendono gli estremi rossi della zona di interesse
    private Boolean Checkestremi(int[] punticontornogial, int[] punticontornoross) {

        if((punticontornogial[0] <= punticontornoross[0]) && (punticontornoross[1] <= punticontornogial[1])){
            if(punticontornogial[3] <= punticontornoross[2]){
                Log.d(TAG, "CONTORNI TROVATI234: " );
                return true;
            }
        }
        return false;
    }

    //controllo se il grigio e presente tra il giallo e il rosso
   /* private Boolean Checkgrey(int[] punticontornogial, int[] punticontornoross, Mat Hsv) {
        double[] colors;
        int
        for(int i=0; i <=   ; i++) {
            colors = Hsv;
        }
    }
*/

    //Funzione che estrae i due punti estremi a x  a y del contorno
    int[] Extractestremi( Point[] contour){
        int[]  returnpoint = new int[5];
        int xmax = 0;
        int xmin = 7000;
        int ymin = 7000;
        int ymax = 0;
        int xdif;
        //SCorro i punti del contorno
        for(int i=0; i< contour.length ; i++){
            int x = (int)contour[i].x;
            int y = (int)contour[i].y;
            if(x < xmin ){xmin = x;}
            if(x > xmax ){xmax = x;}
            if(y < ymin ){ymin = y;}
            if(y > ymax ){ymax = y;}
        }

        xdif = xmax-xmin;

        //PARAMETRO FONDAMENTALE DEL SISTEMA PER OVVIARE ALLA ROTAZIONE
        returnpoint[0]=xmin - (int)(xdif/10);
        returnpoint[1]=xmax + (int)(xdif/10);
        returnpoint[2]=ymin;
        returnpoint[3]=ymax;
        returnpoint[4]=xdif/10;
        return returnpoint;

    }

    //Faccio il check della zona gialla
    private Boolean checkcopyell(Mat mask) {
        double[] color;
        int max = mask.height() *mask.width();
        Log.d(TAG, " lunghezza:" +mask.height());
        Log.d(TAG, " larghezza:" +mask.width());
        int soglia = max/5;
        int num = 0;
        for(int i=0; i < mask.height(); i++) {
            for (int y = 0; y < mask.width(); y++) {
                color = mask.get( i, y);
                Log.d(TAG, " valori:" +Integer.toString((int)color[0]) + Integer.toString((int)color[1]));
                 if( (35 <= (int)color[0]) && ((int)color[0] <= 40) && (55 <= (int)color[1]) ) {
                    num++;
                     if(num > soglia){
                         return true;
                     }
                 }
            }
        }
        return false;

    }

    //Estrazione e comparazione dei descrittori
    Mat Extractdesc(Mat mRgb){//}, Mat filt){

        // Assegno le varie matrici appoggio
        Mat mRgb3 = new Mat();
        keypoints2 = new MatOfKeyPoint();
        descriptortwo = new Mat();
        Mat dst = new Mat();
        KeyPoint tkp[];
        KeyPoint tkp1[];
        Mat element3;


        //Eseguo il Blur dell'immagine
        Imgproc.GaussianBlur( mRgb, dst, new Size(5,5), 2, 2);

        // Trovo i keypoints e estraggo i suoi descrittori
        detector.detect(dst, keypoints2);
        extractor.compute(dst, keypoints2, descriptortwo);

        //Estratti i descrittori gli comparo per trovare i corrispondenti
        MatOfDMatch matches = new MatOfDMatch();
        try {
            matcher.match(descriptorone, descriptortwo, matches);
        }catch(Exception e){
            matches = new MatOfDMatch();
            Log.d(TAG,e.toString());
        }

        //Assegno i keypoint in formato array e anche i matcher point
        tkp = keypoints2.toArray();
        tkp1 = keypoints1.toArray();
        Log.d(TAG, "NUMERO KEYPOINT ESTRATTI:" + Integer.toString(tkp.length)+" e" + Integer.toString(tkp1.length));

        //Assegno elemento di erosione per togliere i punti sugli angoli
        element3 = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(10, 10));
       // Imgproc.erode(filt, filt, element3);

        //Utilizzo queste variabili  per immaganizzare i valori dei colori
        double[] colors ;
        double[] hsv;

        // Ora seleziono solo i punti più validi a seconda delle distanze
        double max_dist = 0;
        double min_dist = 100;
        List<DMatch> matchesList = matches.toList();

        if( !matches.empty() ) {

            //Seleziono la dist minima e massima in base alle distanze dei keypoints date dal matcher
            for (int i = 0; i < descriptorone.rows(); i++) {
                Double distance = (double) matchesList.get(i).distance;
                if (distance < min_dist) min_dist = distance;
                if (distance > max_dist) max_dist = distance;
            }

            //Seleziono i Keypoint del frame della fotocamere a seconda della loro distanza
            for (int i = 0; i < descriptorone.rows(); i++) {
                if (matchesList.get(i).distance < 1.2 * min_dist) {
                    Log.d(TAG, "distanza:" + Double.toString(matchesList.get(i).distance) );
                    Log.d(TAG, "distanza min:" + Double.toString(min_dist) );

                    //Assegno le  coordinate  dei punti Keypoints
                    int x = (int)tkp[matchesList.get(i).trainIdx].pt.x;
                    int y = (int)tkp[matchesList.get(i).trainIdx].pt.y;
                    Log.d(TAG, "Good point:" +  Integer.toString(x) + "y:" + Integer.toString(y));
//                    colors = filt.get(y, x);
//                    hsv = dst.get(y, x);
//                    Log.d(TAG, "valore:" +  Integer.toString((int)colors[0]));

                    //Filtro ulteriormente i Keypoints a seconda se si trovano o no su colori dell'oggetto di interesse
                    //if(((int)colors[0] != 0)){// && ( (34 <= (int)hsv[0]) && ((int)hsv[0] <= 40) )) {
                        Core.circle(mRgb, tkp[matchesList.get(i).trainIdx].pt, 1, new Scalar(0, 255, 0), 2);
                        Log.d(TAG, "Good point");
                    //}
                }
            }
        }

        return mRgb;
    }

    // Utilizzare questa funzione in caso di utilizzo degli occhiali sull'occhio sinistro per ruotare il frame
    Mat rotateImage( Mat source, double angle)
    {
        Mat dst = new Mat();
        Log.d(TAG,"Grandezza matroice" + source.size().toString());
        Mat rot_mat = Imgproc.getRotationMatrix2D(new Point(source.cols()/2.0F, source.rows()/2.0F), angle, 1.0);
        Imgproc.warpAffine(source, dst, rot_mat, source.size());
        return dst;
    }

    //Calcolo Histogramma dell' immagine selezionata
    private Mat calHist(Mat src){
        Mat results = new Mat();
        Mat mH = new Mat();
        Mat mS = new Mat();
        Mat mV = new Mat();
        Mat mhsv= new Mat();

        ///Imgproc.cvtColor(src, mRgb , Imgproc.COLOR_RGB2HSV_FULL, 4);
        List<Mat>  channels = new ArrayList<Mat>(2);

        Imgproc.cvtColor(src, mhsv, Imgproc.COLOR_RGB2HSV_FULL, 3);

        List<Mat> arrMat = Arrays.asList(mhsv);

        MatOfInt channel = new MatOfInt(0);

        Mat hist= new Mat();
       MatOfInt histSize = new MatOfInt(50);
        MatOfFloat ranges = new MatOfFloat(0f, 256f);
        Imgproc.calcHist(arrMat, channel, new Mat(), hist, histSize, ranges);
        for (int i = 0; i< 50; i++) {
            double[] histValues = hist.get(i, 0);
            for (int j = 0; j < histValues.length; j++) {
                Log.d(TAG, "H1=" + histValues[j]);
            }
        }

        channel = new MatOfInt(1);
        Imgproc.calcHist(arrMat, channel, new Mat(), hist, histSize, ranges);
        for (int i = 0; i< 50; i++) {
            double[] histValues = hist.get(i, 0);
            for (int j = 0; j < histValues.length; j++) {
                Log.d(TAG, "S1=" + histValues[j]);
            }
        }

        return hist;
    }


    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                    //detector = FeatureDetector.create(FeatureDetector.SIFT);
                    //extractor = DescriptorExtractor.create(DescriptorExtractor.SURF);
                    //matcher = DescriptorMatcher.create(DescriptorMatcher.FLANNBASED);
                    break;

                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };

}