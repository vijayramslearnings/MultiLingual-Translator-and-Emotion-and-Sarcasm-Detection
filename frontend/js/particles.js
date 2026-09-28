/**
 * Bhasha Live — Interactive 3D Particle Orbit System
 * Features:
 *   - 3D Spherical point cloud with perspective projection
 *   - Constellation proximity connections
 *   - Mouse cursor drag and inertia tracking
 *   - Real-time audio amplitude expansion & neon pulse
 */

(function () {
  const canvas = document.getElementById("particleOrbitCanvas");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  let width, height, dpr;

  // Particle Settings
  const NUM_PARTICLES = 160;
  const BASE_RADIUS = 125;
  let dynamicRadius = BASE_RADIUS;
  let targetRadius = BASE_RADIUS;
  const FOV = 350;

  // Rotation & Physics
  let angleX = 0.2;
  let angleY = 0.2;
  let targetAngleX = 0.2;
  let targetAngleY = 0.2;
  let rotSpeedX = 0.003;
  let rotSpeedY = 0.005;

  let isHovered = false;
  let mouseX = 0, mouseY = 0;
  let audioAmplitude = 0.0; // 0.0 to 1.0

  class Point3D {
    constructor() {
      // Golden spiral distribution on sphere surface
      this.theta = Math.acos(2 * Math.random() - 1);
      this.phi = 2 * Math.PI * Math.random();
      this.baseSize = 1.5 + Math.random() * 2.2;
      this.size = this.baseSize;
      
      // Color tint
      const tints = ["#6366f1", "#8b5cf6", "#06b6d4", "#a5b4fc"];
      this.color = tints[Math.floor(Math.random() * tints.length)];
    }

    getPosition(radius) {
      const x = radius * Math.sin(this.theta) * Math.cos(this.phi);
      const y = radius * Math.sin(this.theta) * Math.sin(this.phi);
      const z = radius * Math.cos(this.theta);
      return { x, y, z };
    }
  }

  const particles = [];
  for (let i = 0; i < NUM_PARTICLES; i++) {
    particles.push(new Point3D());
  }

  function resize() {
    dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    width = rect.width;
    height = rect.height;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
  }

  window.addEventListener("resize", resize);
  resize();

  // Mouse Interactivity
  canvas.addEventListener("mousemove", (e) => {
    const rect = canvas.getBoundingClientRect();
    mouseX = (e.clientX - rect.left) / width - 0.5;
    mouseY = (e.clientY - rect.top) / height - 0.5;
    targetAngleY = mouseX * 2.5;
    targetAngleX = -mouseY * 2.5;
    isHovered = true;
  });

  canvas.addEventListener("mouseleave", () => {
    isHovered = false;
  });

  // Public API to hook into audio amplitude
  window.setParticleAudioAmplitude = function (amp) {
    audioAmplitude = Math.max(0, Math.min(1.0, amp));
    targetRadius = BASE_RADIUS + audioAmplitude * 55;
  };

  // 3D Matrix Rotation (Euler)
  function rotate3D(p, rx, ry) {
    // Rotate Y
    const cosY = Math.cos(ry);
    const sinY = Math.sin(ry);
    let x1 = p.x * cosY + p.z * sinY;
    let z1 = -p.x * sinY + p.z * cosY;

    // Rotate X
    const cosX = Math.cos(rx);
    const sinX = Math.sin(rx);
    let y2 = p.y * cosX - z1 * sinX;
    let z2 = p.y * sinX + z1 * cosX;

    return { x: x1, y: y2, z: z2 };
  }

  // Animation Loop
  function render() {
    ctx.clearRect(0, 0, width, height);

    // Smooth inertia
    if (isHovered) {
      angleX += (targetAngleX - angleX) * 0.08;
      angleY += (targetAngleY - angleY) * 0.08;
    } else {
      angleX += rotSpeedX;
      angleY += rotSpeedY + audioAmplitude * 0.01;
    }

    dynamicRadius += (targetRadius - dynamicRadius) * 0.12;

    const projectedPoints = [];
    const cx = width / 2;
    const cy = height / 2;

    // Project points to 2D
    for (let i = 0; i < particles.length; i++) {
      const pos = particles[i].getPosition(dynamicRadius);
      const rot = rotate3D(pos, angleX, angleY);

      const scale = FOV / (FOV + rot.z + dynamicRadius);
      const px = cx + rot.x * scale;
      const py = cy + rot.y * scale;
      const alpha = Math.max(0.15, (rot.z + dynamicRadius) / (dynamicRadius * 2));

      projectedPoints.push({
        x: px,
        y: py,
        z: rot.z,
        scale: scale,
        alpha: alpha,
        color: particles[i].color,
        size: particles[i].baseSize * scale * (1 + audioAmplitude * 0.8)
      });
    }

    // Sort by depth (painter's algorithm)
    projectedPoints.sort((a, b) => a.z - b.z);

    // Draw Constellation Lines between close neighbors
    ctx.lineWidth = 0.75;
    for (let i = 0; i < projectedPoints.length; i++) {
      for (let j = i + 1; j < projectedPoints.length; j++) {
        const p1 = projectedPoints[i];
        const p2 = projectedPoints[j];
        const dist = Math.hypot(p1.x - p2.x, p1.y - p2.y);

        if (dist < 48) {
          const lineAlpha = (1 - dist / 48) * p1.alpha * 0.35;
          ctx.strokeStyle = `rgba(99, 102, 241, ${lineAlpha})`;
          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.stroke();
        }
      }
    }

    // Draw Particle Nodes with Glow
    for (let i = 0; i < projectedPoints.length; i++) {
      const p = projectedPoints[i];
      ctx.save();
      ctx.globalAlpha = p.alpha;
      ctx.fillStyle = p.color;

      // Glow halo
      ctx.shadowBlur = 10 * (1 + audioAmplitude);
      ctx.shadowColor = p.color;

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    requestAnimationFrame(render);
  }

  requestAnimationFrame(render);
})();
