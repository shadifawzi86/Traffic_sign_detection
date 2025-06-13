import argparse
import cv2
import time
from hybrid_system import HybridSignSystem 

def main():
    parser = argparse.ArgumentParser(description="Traffic Sign Recognition System")
    subparsers = parser.add_subparsers(dest='command', required=True)

    
    train_parser = subparsers.add_parser('train', help='Train classifier')
    train_parser.add_argument('--dataset-path', required=True)
    train_parser.add_argument('--epochs', type=int, default=30)
    train_parser.add_argument('--batch-size', type=int, default=32)
    train_parser.add_argument('--output', default='my_classifier.pth')

    # Run command 
    run_parser = subparsers.add_parser('run', help='Run inference')
    run_parser.add_argument('--detector', default='best_1.pt')
    run_parser.add_argument('--classifier', default='classifier.pth')
    run_parser.add_argument('--source', type=str, default=r"D:\traffic_sign_system\v.mp4")
    run_parser.add_argument('--show-fps', action='store_true')

    args = parser.parse_args()

    if args.command == 'train':
        from train_classifier import train
        train(
            dataset_path=args.dataset_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            output=args.output
        )
        
    elif args.command == 'run':
        system = HybridSignSystem(args.detector, args.classifier)
        cap = cv2.VideoCapture(args.source)
        
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            results = system.process_frame(frame)
            
            # Draw results
            for obj in results:
                x1, y1, x2, y2 = obj['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{obj.get('class_name', 'Unknown')} ({obj.get('confidence', 0):.2f})"

                #label = f"{obj['class_name']} ({obj['confidence']:.2f})"
                cv2.putText(frame, label, (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if args.show_fps:
                fps = 1 / (time.time() - start_time)
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Traffic Sign Recognition', frame)
            if cv2.waitKey(1) == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
