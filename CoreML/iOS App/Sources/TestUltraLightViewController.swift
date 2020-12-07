//
//  TestYOLOViewController.swift
//  CS498
//
//  Created by Minghao Jiang on 11/27/20.
//  Copyright © 2020 MachineThink. All rights reserved.
//

import UIKit
import CoreML
import Vision

/**
  View controller for the "Camera" screen.

  This can evaluate both the neural network model and the k-NN model, because
  they use the same input and output names.

  On the Simulator this uses the photo library instead of the camera.
 */
class TestUltraLightViewController: UIViewController {
  @IBOutlet var cameraButton: UIBarButtonItem!
  @IBOutlet var imageView: UIImageView!
  @IBOutlet var textView: UITextView!

  var predictor: UltraLightFaceNet!
    
    var boundingBoxes = [BoundingBox]()
    var colors: [UIColor] = []

  override func viewDidLoad() {
    super.viewDidLoad()
    textView.text = "Use the camera to take a photo."
    predictor = UltraLightFaceNet()
    
    setUpBoundingBoxes()
  }
    
    let labels = [
      "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
      "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
      "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]

  @IBAction func takePicture() {
    let picker = UIImagePickerController()
    picker.delegate = self
    picker.sourceType = UIImagePickerController.isSourceTypeAvailable(.camera) ? .camera : .photoLibrary
    picker.allowsEditing = true
    present(picker, animated: true)
  }
    
    func setUpBoundingBoxes() {
      for _ in 0..<YOLO.maxBoundingBoxes {
        boundingBoxes.append(BoundingBox())
      }

      // Make colors for the bounding boxes. There is one color for each class,
      // 20 classes in total.
      for r: CGFloat in [0.2, 0.4, 0.6, 0.8, 1.0] {
        for g: CGFloat in [0.3, 0.7] {
          for b: CGFloat in [0.4, 0.8] {
            let color = UIColor(red: r, green: g, blue: b, alpha: 1)
            colors.append(color)
          }
        }
      }
        
        for box in self.boundingBoxes {
          box.addToLayer(self.imageView.layer)
        }
    }

  func predict(image: UIImage) {
    // In the "Evaluate" screen we load the image from the dataset but here
    // we use a UIImage / CGImage object.
    if let pixelBuffer = image.pixelBuffer(width: 224, height: 224) {
        if let predictions = try? predictor.prediction(input: pixelBuffer) {
            print(predictions)
//            textView.text = makeDisplayString(predictions)
        }
        else {
            
        }
        
        
//        for i in 0..<boundingBoxes.count {
//          if i < predictions.count {
//            let prediction = predictions[i]
//
//            // The predicted bounding box is in the coordinate space of the input
//            // image, which is a square image of 416x416 pixels. We want to show it
//            // on the video preview, which is as wide as the screen and has a 16:9
//            // aspect ratio. The video preview also may be letterboxed at the top
//            // and bottom.
//            let height = imageView.bounds.height
//            let width = imageView.bounds.width
//            let scaleX = width / CGFloat(YOLO.inputWidth)
//            let scaleY = height / CGFloat(YOLO.inputHeight)
//            let top = (imageView.bounds.height - height) / 2
//
//            // Translate and scale the rectangle to our own coordinate system.
//            var rect = prediction.rect
//            rect.origin.x *= scaleX
//            rect.origin.y *= scaleY
//            rect.origin.y += top
//            rect.size.width *= scaleX
//            rect.size.height *= scaleY
//
//            // Show the bounding box.
//            let label = String(format: "%@ %.1f", labels[prediction.classIndex], prediction.score * 100)
//
//            print(label)
//
//            let color = colors[prediction.classIndex]
//            if labels[prediction.classIndex] == "person"{
//                boundingBoxes[i].show(frame: rect, label: label, color: color)
//            }
//            else {
//                boundingBoxes[i].hide()
//            }
//          } else {
//            boundingBoxes[i].hide()
//          }
//        }
    }
  }

private func makeDisplayString(_ prediction: [YOLO.Prediction]) -> String {
    var str = ""
    for p in prediction {
        var s = "Prediction: \(labels[p.classIndex])\n"
        s += String(format: "Probability: %.2f%%\n\n", p.score * 100)
        s += "\n"
        str += s
    }
    return str
    }
}

extension TestUltraLightViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
  func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
    picker.dismiss(animated: true)

    let image = info[UIImagePickerController.InfoKey.editedImage] as! UIImage
    imageView.image = image

    predict(image: image)
  }
}


