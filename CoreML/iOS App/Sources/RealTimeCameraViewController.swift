//
//  RealtimeCameraViewController.swift
//  CS498
//
//  Created by Minghao Jiang on 11/14/20.
//  Copyright © 2020 MachineThink. All rights reserved.
//

import UIKit
import ARKit
import AVKit
import Vision
import CoreML
import SceneKit


class RealTimeCameraViewController: UIViewController {
    let captureSession = AVCaptureSession()
    var mlmodel: MLModel!
    
    @IBOutlet weak var cameraView: UIView!
    @IBOutlet weak var nameTag: UILabel!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        self.startingTheCam()
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    func startingTheCam(){

        //Set session preset
        captureSession.sessionPreset = .photo
        
        //Capturing Device
        guard let capturingDevice = AVCaptureDevice.default(for: .video) else { return }
        
        //Capture Input
        guard let capturingInput = try? AVCaptureDeviceInput(device: capturingDevice) else { return }
        
        //Adding input to capture session
        captureSession.addInput(capturingInput)

        //Data output
        let cameraDataOutput = AVCaptureVideoDataOutput()
        cameraDataOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "outputVideo"))
        captureSession.addOutput(cameraDataOutput)
        
        //Construct a camera preview layer
        let cameraPreviewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        
        //Set the frame
        cameraPreviewLayer.frame = cameraView.bounds
        
        //Add this preview layer to sublayer of view
        cameraView.layer.addSublayer(cameraPreviewLayer)
        
        //Start the session
        captureSession.startRunning()
        
        
    }
    
}

extension RealTimeCameraViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        
        guard let model = try? VNCoreMLModel(for: self.mlmodel) else {
            fatalError("Unable to load model!")
        }
        
        let coremlRequest = VNCoreMLRequest(model: model) { [weak self] request, error in
            
            guard let results = request.results as? [VNClassificationObservation],
                  let topResult = results.first else {
                fatalError("Unexpected results")
            }
            
            DispatchQueue.main.async { [weak self] in
                
                if topResult.identifier != "Unknown"{
                    self?.nameTag.text? = topResult.identifier
                }
            }
        }
        
        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:]).perform([coremlRequest])
    }
}
