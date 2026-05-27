import { useEffect, useRef, useState } from "react";
import {
  Renderer,
  Camera,
  Transform,
  Geometry,
  Program,
  Mesh,
} from "ogl";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";

interface PointCloudViewerProps {
  plyUrl: string;
  isDarkMode?: boolean;
  onLoadComplete?: () => void;
  onError?: (error: Error) => void;
}

export default function PointCloudViewer({
  plyUrl,
  isDarkMode = true,
  onLoadComplete,
  onError,
}: PointCloudViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  const vertexShader = /* glsl */ `
    precision highp float;

    attribute vec3 position;
    attribute vec3 color;

    uniform mat4 modelViewMatrix;
    uniform mat4 projectionMatrix;
    uniform float pointSize;

    varying vec3 vColor;

    void main() {
      vColor = color;
      vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
      gl_Position = projectionMatrix * mvPosition;
      gl_PointSize = pointSize * (300.0 / -mvPosition.z);
    }
  `;

  const fragmentShader = /* glsl */ `
    precision highp float;

    varying vec3 vColor;

    void main() {
      // Circular point shape
      vec2 center = gl_PointCoord - vec2(0.5);
      float dist = length(center);
      if (dist > 0.5) discard;

      // Smooth edges
      float alpha = 1.0 - smoothstep(0.4, 0.5, dist);

      gl_FragColor = vec4(vColor, alpha);
    }
  `;

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // Check WebGL support
    const canvas = document.createElement("canvas");
    const gl =
      canvas.getContext("webgl") || canvas.getContext("experimental-webgl");
    if (!gl) {
      const error = new Error("WebGL not supported");
      setLoadError("WebGL is not supported in your browser");
      onError?.(error);
      return;
    }

    // Render-loop handle, hoisted so the effect cleanup can stop it even
    // though it is started later inside the async PLY load callback.
    let rafId = 0;
    let disposed = false;

    const renderer = new Renderer({ alpha: true, antialias: true });
    const rendererGl = renderer.gl;
    rendererGl.clearColor(0, 0, 0, 0);
    container.appendChild(rendererGl.canvas);

    const camera = new Camera(rendererGl, { fov: 45 });
    camera.position.set(0, 0, 5);

    const scene = new Transform();

    // Camera control state
    let cameraDistance = 5;
    let cameraRotationX = 0;
    let cameraRotationY = 0;
    let cameraPanX = 0;
    let cameraPanY = 0;

    let isDragging = false;
    let isPanning = false;
    let lastMouseX = 0;
    let lastMouseY = 0;

    // Touch state
    let lastTouchDistance = 0;

    function resize() {
      if (!container) return;
      const dpr = window.devicePixelRatio || 1;
      const width = container.clientWidth;
      const height = container.clientHeight;
      renderer.setSize(width * dpr, height * dpr);
      rendererGl.canvas.style.width = width + "px";
      rendererGl.canvas.style.height = height + "px";
      camera.perspective({ aspect: width / height });
    }

    window.addEventListener("resize", resize);
    resize();

    function updateCamera() {
      const x =
        cameraDistance *
        Math.sin(cameraRotationY) *
        Math.cos(cameraRotationX);
      const y = cameraDistance * Math.sin(cameraRotationX);
      const z =
        cameraDistance *
        Math.cos(cameraRotationY) *
        Math.cos(cameraRotationX);

      camera.position.set(x + cameraPanX, y + cameraPanY, z);
      camera.lookAt([cameraPanX, cameraPanY, 0]);
    }

    // Mouse controls
    const handleMouseDown = (e: MouseEvent) => {
      if (e.ctrlKey || e.metaKey) {
        isPanning = true;
      } else {
        isDragging = true;
      }
      lastMouseX = e.clientX;
      lastMouseY = e.clientY;
      e.preventDefault();
    };

    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging && !isPanning) return;

      const deltaX = e.clientX - lastMouseX;
      const deltaY = e.clientY - lastMouseY;
      lastMouseX = e.clientX;
      lastMouseY = e.clientY;

      if (isPanning) {
        const panSpeed = 0.005 * cameraDistance;
        cameraPanX -= deltaX * panSpeed;
        cameraPanY += deltaY * panSpeed;
      } else if (isDragging) {
        cameraRotationY += deltaX * 0.01;
        cameraRotationX -= deltaY * 0.01;
        cameraRotationX = Math.max(
          -Math.PI / 2,
          Math.min(Math.PI / 2, cameraRotationX),
        );
      }

      updateCamera();
    };

    const handleMouseUp = () => {
      isDragging = false;
      isPanning = false;
    };

    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();
      cameraDistance *= 1 + e.deltaY * 0.001;
      cameraDistance = Math.max(0.5, Math.min(50, cameraDistance));
      updateCamera();
    };

    // Touch controls
    const handleTouchStart = (e: TouchEvent) => {
      if (e.touches.length === 1) {
        isDragging = true;
        lastMouseX = e.touches[0].clientX;
        lastMouseY = e.touches[0].clientY;
      } else if (e.touches.length === 2) {
        isDragging = false;
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dy = e.touches[0].clientY - e.touches[1].clientY;
        lastTouchDistance = Math.sqrt(dx * dx + dy * dy);
      }
      e.preventDefault();
    };

    const handleTouchMove = (e: TouchEvent) => {
      if (e.touches.length === 1 && isDragging) {
        const deltaX = e.touches[0].clientX - lastMouseX;
        const deltaY = e.touches[0].clientY - lastMouseY;
        lastMouseX = e.touches[0].clientX;
        lastMouseY = e.touches[0].clientY;

        cameraRotationY += deltaX * 0.01;
        cameraRotationX -= deltaY * 0.01;
        cameraRotationX = Math.max(
          -Math.PI / 2,
          Math.min(Math.PI / 2, cameraRotationX),
        );
        updateCamera();
      } else if (e.touches.length === 2) {
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dy = e.touches[0].clientY - e.touches[1].clientY;
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (lastTouchDistance > 0) {
          const delta = distance - lastTouchDistance;
          cameraDistance *= 1 - delta * 0.01;
          cameraDistance = Math.max(0.5, Math.min(50, cameraDistance));
          updateCamera();
        }

        lastTouchDistance = distance;
      }
      e.preventDefault();
    };

    const handleTouchEnd = () => {
      isDragging = false;
      lastTouchDistance = 0;
    };

    container.addEventListener("mousedown", handleMouseDown);
    container.addEventListener("mousemove", handleMouseMove);
    container.addEventListener("mouseup", handleMouseUp);
    container.addEventListener("mouseleave", handleMouseUp);
    container.addEventListener("wheel", handleWheel, { passive: false });
    container.addEventListener("touchstart", handleTouchStart, {
      passive: false,
    });
    container.addEventListener("touchmove", handleTouchMove, {
      passive: false,
    });
    container.addEventListener("touchend", handleTouchEnd);

    // Load PLY file
    const loader = new PLYLoader();
    loader.load(
      plyUrl,
      (geometry) => {
        try {
          const positions = geometry.getAttribute("position");
          const colors = geometry.getAttribute("color");

          if (!positions) {
            throw new Error("PLY file has no position data");
          }

          const posArray = new Float32Array(positions.array);
          const colorArray = colors
            ? new Float32Array(colors.array)
            : new Float32Array(posArray.length).fill(1.0); // White if no colors

          // Normalize colors if they're in 0-255 range
          if (colors && colors.array[0] > 1) {
            for (let i = 0; i < colorArray.length; i++) {
              colorArray[i] /= 255;
            }
          }

          // Calculate bounding box for auto-fit camera
          let minX = Infinity,
            minY = Infinity,
            minZ = Infinity;
          let maxX = -Infinity,
            maxY = -Infinity,
            maxZ = -Infinity;

          for (let i = 0; i < posArray.length; i += 3) {
            minX = Math.min(minX, posArray[i]);
            minY = Math.min(minY, posArray[i + 1]);
            minZ = Math.min(minZ, posArray[i + 2]);
            maxX = Math.max(maxX, posArray[i]);
            maxY = Math.max(maxY, posArray[i + 1]);
            maxZ = Math.max(maxZ, posArray[i + 2]);
          }

          const centerX = (minX + maxX) / 2;
          const centerY = (minY + maxY) / 2;
          const sizeX = maxX - minX;
          const sizeY = maxY - minY;
          const sizeZ = maxZ - minZ;
          const maxSize = Math.max(sizeX, sizeY, sizeZ);

          // Auto-fit camera (pulled back a little so it's easy to orbit)
          cameraDistance = maxSize * 2.0;
          cameraPanX = centerX;
          cameraPanY = centerY;
          updateCamera();

          // Create OGL geometry
          const oglGeometry = new Geometry(rendererGl, {
            position: { size: 3, data: posArray },
            color: { size: 3, data: colorArray },
          });

          const program = new Program(rendererGl, {
            vertex: vertexShader,
            fragment: fragmentShader,
            uniforms: {
              // Scale point size to the model: the shader applies a 300/-z
              // perspective factor, so for unit-normalized clouds a fixed
              // value of 2.0 yields ~400px blobs. Tying it to maxSize keeps
              // points ~6 device-px at the default fit, at any model scale.
              pointSize: { value: maxSize * 0.03 },
            },
            transparent: true,
            depthTest: true,
            depthWrite: false,
          });

          const points = new Mesh(rendererGl, {
            mode: rendererGl.POINTS,
            geometry: oglGeometry,
            program: program,
          });

          points.setParent(scene);

          setIsLoading(false);
          onLoadComplete?.();

          // Render loop
          const render = () => {
            if (disposed) return;
            rafId = requestAnimationFrame(render);
            renderer.render({ scene, camera });
          };
          rafId = requestAnimationFrame(render);
        } catch (err) {
          const error =
            err instanceof Error
              ? err
              : new Error("Failed to process PLY file");
          console.error("Error processing PLY:", error);
          setLoadError(error.message);
          setIsLoading(false);
          onError?.(error);
        }
      },
      undefined,
      (error) => {
        console.error("Error loading PLY:", error);
        setLoadError("Failed to load point cloud file");
        setIsLoading(false);
        onError?.(
          error instanceof Error ? error : new Error("Failed to load PLY"),
        );
      },
    );

    return () => {
      disposed = true;
      cancelAnimationFrame(rafId);
      window.removeEventListener("resize", resize);
      container.removeEventListener("mousedown", handleMouseDown);
      container.removeEventListener("mousemove", handleMouseMove);
      container.removeEventListener("mouseup", handleMouseUp);
      container.removeEventListener("mouseleave", handleMouseUp);
      container.removeEventListener("wheel", handleWheel);
      container.removeEventListener("touchstart", handleTouchStart);
      container.removeEventListener("touchmove", handleTouchMove);
      container.removeEventListener("touchend", handleTouchEnd);
      container.removeChild(rendererGl.canvas);
      rendererGl.getExtension("WEBGL_lose_context")?.loseContext();
    };
  }, [plyUrl, isDarkMode, onLoadComplete, onError]);

  return (
    <div className="relative w-full h-full">
      <div ref={containerRef} className="w-full h-full" />
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-background/80 backdrop-blur-sm">
          <div className="text-center space-y-2">
            <div className="animate-spin h-8 w-8 border-4 border-primary border-t-transparent rounded-full mx-auto" />
            <p className="text-sm text-muted-foreground">
              Loading point cloud...
            </p>
          </div>
        </div>
      )}
      {loadError && (
        <div className="absolute inset-0 flex items-center justify-center bg-background/80 backdrop-blur-sm">
          <div className="text-center space-y-2 p-4">
            <p className="text-sm text-destructive">{loadError}</p>
            <p className="text-xs text-muted-foreground">
              Please try refreshing the page
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
