## XADE Data Flow Sequence

Mobile/Desktop Client -> FastAPI Server -> Detection -> Explanation -> Response

1. Client sends image (multipart/form-data)
2. Server validates and saves image
3. Detection model processes image
4. GradCAM generates heatmap
5. VLM provider generates explanation
6. Results saved to database
7. Resposne returned to client
8. Client displays results