package com.example.bocektespiti;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.bocektespiti.ml.Model;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    TextView result, confidence;
    ImageView imageView;
    Button camera,gallery;
    int imageSize = 200;

    @SuppressLint("MissingInflatedId")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        result = findViewById(R.id.result);
        confidence = findViewById(R.id.confidence);
        imageView = findViewById(R.id.imageView);
        camera = findViewById(R.id.cameraButton);
        gallery = findViewById(R.id.galleryButton);


        camera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Launch camera if we have permission
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                    if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                        Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                        startActivityForResult(cameraIntent, 1);
                    } else {
                        //Request camera permission if we don't have it.
                        requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                    }
                }
            }
        });

        gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent galeryIntent = new Intent(Intent.ACTION_GET_CONTENT);
                galeryIntent.setType("image/*");
                startActivityForResult(galeryIntent,2);
            }
        });
    }

    public void classifyImage(Bitmap image){
        try {
            Model model = Model.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 200, 200, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4*imageSize*imageSize*3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intvalues = new int[imageSize*imageSize];
            image.getPixels(intvalues,0,image.getWidth(),0,0,image.getWidth(),image.getHeight());
            int pixel=0;
            for (int i=0; i<imageSize; i++){
                for (int j=0; j<imageSize; j++){
                    int val = intvalues[pixel++];
                    byteBuffer.putFloat(((val>>16)&0xFF)*(1.f/255.f));
                    byteBuffer.putFloat(((val>>8)&0xFF)*(1.f/255.f));
                    byteBuffer.putFloat((val &0xFF)*(1.f/255.f));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidences.length; i++){
                if (confidences[i] > maxConfidence){
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }

            String classes[] = {"paddy stem maggot","yellow rice borer ","rice gall midge","Rice Stemfly ","brown plant hopper",
                    "rice leafhopper","grain spreader thrips","rice shell pest","mole cricket","wireworm","white margined moth",
                    "red spider","Potosiabre vitarsis","wheat blossom midge","penthaleus major","longlegged spider mite","wheat phloeothrips",
                    "wheat sawfly","cerodonta denticornis","beet fly","Beet spot flies","meadow moth","beet weevil","sericaorient alismots chulsky",
                    "alfalfa weevil","tarnished plant bug","Locustoidea","therioaphis maculata Buckton","odontothrips loti","alfalfa seed chalcid",
                    "Pieris canidia","Apolygus lucorum","Viteus vitifoliae","Colomerus vitis","Brevipoalpus lewisi McGregor",
                    "oides decempunctata","Polyphagotars onemus latus","Pseudococcus comstocki Kuwana","parathrene regalis",
                    "Ampelophaga","Trialeurodes vaporariorum","Erythroneura apicalis","Papilio xuthus","Panonchus citri McGregor",
                    "Phyllocoptes oleiverus ashmead","Icerya purchasi Maskell","Unaspis yanonensis","Ceroplastes rubens","Chrysomphalus aonidum",
                    "Parlatoria zizyphus Lucus","Nipaecoccus vastalor","Aleurocanthus spiniferus","Tetradacus c Bactrocera minax","Dacus dorsalis(Hendel)",
                    "Bactrocera tsuneonis","Adristyrannus","Phyllocnistis citrella Stainton","Toxoptera citricidus","Toxoptera aurantii",
                    "Aphis citricola Vander Goot","Dasineura sp","Lawana imitata Melichar","Salurnis marginella Guerr","Deporaus marginatus Pascoe",
                    "Chlumetia transversa","Mango flat beak leafhopper",};

            result.setText(classes[maxPos]);

            /*String s = "";
            for (int i = 0; i< 10; i++){
                s += String.format("%s: %.1f%%\n",classes[i],confidences[i]*100);
            }
            confidence.setText(s);*/

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }

    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (requestCode == 1 && resultCode == RESULT_OK) {
            Bitmap image = (Bitmap) data.getExtras().get("data");
            int dimension = Math.min(image.getWidth(),image.getHeight());
            image = ThumbnailUtils.extractThumbnail(image,dimension,dimension);
            imageView.setImageBitmap(image);

            image=Bitmap.createScaledBitmap(image,imageSize,imageSize,true);
            classifyImage(image);
        }
        else if (requestCode == 2){
            if (data != null){
                Uri uri = data.getData();
                try {
                    Bitmap image = MediaStore.Images.Media.getBitmap(this.getContentResolver(),uri);
                    int dimension = Math.min(image.getWidth(),image.getHeight());
                    image = ThumbnailUtils.extractThumbnail(image,dimension,dimension);
                    imageView.setImageBitmap(image);

                    image=Bitmap.createScaledBitmap(image,imageSize,imageSize,true);
                    classifyImage(image);
                }
                catch (IOException e){
                    e.printStackTrace();
                }
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
}