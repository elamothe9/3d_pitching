import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";

const API = "http://127.0.0.1:8000"; // FastAPI backend
const SEASON_START = "2020-03-01";
const SEASON_END   = "2024-11-30";


const TYPE_NAME = {
  FF: "Four-seam Fastball",
  FT: "Two-seam Fastball",
  SI: "Sinker",
  FC: "Cutter",
  SL: "Slider",
  CU: "Curveball",
  KC: "Knuckle Curve",
  CH: "Changeup",
  ST: "Sweeper",
  SW: "Sweeper",
  FS: "Splitter",
  EP: "Eephus",
  KN: "Knuckleball",
};

const TYPE_COLOR = {
  FF: 0xff5555,
  FT: 0xff8855,
  SI: 0xffaa33,
  FC: 0xffcc66,
  SL: 0x66aaff,
  CU: 0x44ccff,
  KC: 0x33eeee,
  CH: 0x77dd77,
  ST: 0x9d7dff,
  SW: 0x9d7dff,
  FS: 0xdddd66,
  EP: 0xffffff,
  KN: 0xffffff,
};

async function safeJson(res) {
  try {
    return await res.json();
  } catch {
    return null;
  }
}


// ───────────────────────── Physics helpers (ft, s, slug) ─────────────────────────
const G_VEC = new THREE.Vector3(0, 0, -32.174);
const RHO   = 0.0023769;   // sea level
const R_B   = 0.1205;      // baseball radius (ft)
const A_B   = Math.PI * R_B * R_B;
const M_B   = 0.320;       // slugs

const mph      = v_ft_s => v_ft_s * 0.681818;
const rpm2radS = rpm => rpm * 2 * Math.PI / 60;

function CD_from_speed(vmag, cd0=0.33, cd1=0.02) {
  const cd = cd0 + cd1 * (90 - mph(vmag)) / 20;
  return Math.min(0.45, Math.max(0.25, cd));
}
function CL_from_S(S, k1=0.40, k2=0.35) {
  return (k1 * S) / (1 + k2 * S);
}
function transverseSpin(omega, v) {
  const vmag = v.length();
  if (vmag < 1e-9) return { omThat: new THREE.Vector3(0,0,1), omTmag: 0 };
  const vhat  = v.clone().multiplyScalar(1/vmag);
  const proj  = vhat.clone().multiplyScalar(omega.dot(vhat));
  const omegaT= omega.clone().sub(proj);
  const m     = omegaT.length();
  return { omThat: m > 1e-9 ? omegaT.clone().multiplyScalar(1/m) : new THREE.Vector3(0,0,1), omTmag: m };
}
function estimateSpinAxisFromAccel(ax, az, v0) {
  const aLat = new THREE.Vector3(ax || 0, 0, az || 0);
  const dir  = v0.clone().cross(aLat);
  if (dir.length() < 1e-9) return new THREE.Vector3(0,0,1);
  return dir.normalize();
}
function rotationFromYawPitch(yaw, pitch) {
  const Ry = new THREE.Matrix4().makeRotationY(yaw);
  const Rz = new THREE.Matrix4().makeRotationZ(pitch);
  return new THREE.Matrix4().multiplyMatrices(Ry, Rz);
}
function rk4Step(r, v, dt, omega, cdParams, clParams) {
  const accel = (vel) => {
    const vmag = vel.length();
    const CD   = CD_from_speed(vmag, ...cdParams);
    const Kd   = 0.5 * RHO * CD * A_B / M_B;

    const { omThat, omTmag } = transverseSpin(omega, vel);
    const S    = (R_B * omTmag) / (vmag + 1e-9);
    const CL   = CL_from_S(S, ...clParams);
    const Km   = 0.5 * RHO * CL * A_B / M_B;

    const aDrag   = vel.clone().multiplyScalar(-Kd * vmag);
    const aMagnus = omThat.clone().cross(vel).multiplyScalar(Km * vmag);
    return aDrag.add(aMagnus).add(G_VEC);
  };


  const k1v = accel(v.clone());        const k1r = v.clone();
  const v2  = v.clone().addScaledVector(k1v, dt*0.5);
  const k2v = accel(v2);               const k2r = v.clone().addScaledVector(k1v, dt*0.5);
  const v3  = v.clone().addScaledVector(k2v, dt*0.5);
  const k3v = accel(v3);               const k3r = v.clone().addScaledVector(k2v, dt*0.5);
  const v4  = v.clone().addScaledVector(k3v, dt);
  const k4v = accel(v4);               const k4r = v.clone().addScaledVector(k3v, dt);

  const vNew = v.clone().addScaledVector(k1v, dt/6).addScaledVector(k2v, dt/3).addScaledVector(k3v, dt/3).addScaledVector(k4v, dt/6);
  const rNew = r.clone().addScaledVector(k1r, dt/6).addScaledVector(k2r, dt/3).addScaledVector(k3r, dt/3).addScaledVector(k4r, dt/6);
  return { rNew, vNew };
}
function simulateToPlate(r0, v0, omega, {
  dt = 1/1000, plateY = 0, maxT = 1.2,
  cdParams = [0.33, 0.02], clParams = [0.40, 0.35],
  aSSW_body = null, R_body = null,
  sample = true
} = {}) {
  let r = r0.clone(), v = v0.clone(), t = 0;
  const path = sample ? [{ t, x: r.x, y: r.y, z: r.z }] : null;

  while (t < maxT && r.y > plateY) {
    const { rNew, vNew } = rk4Step(r, v, dt, omega, cdParams, clParams);
    if (aSSW_body && R_body) {
      const aWorld = aSSW_body.clone().applyMatrix4(R_body);
      vNew.addScaledVector(aWorld, dt);
      rNew.addScaledVector(aWorld, 0.5 * dt * dt);
    }
    r.copy(rNew); v.copy(vNew); t += dt;
    if (sample) path.push({ t, x: r.x, y: r.y, z: r.z });
  }

  if (r.y < plateY && sample && path.length >= 2) {
    const p1 = path[path.length - 2], p2 = path[path.length - 1];
    const s = (p1.y - plateY) / ((p1.y - p2.y) + 1e-9);
    const xi = p1.x + s * (p2.x - p1.x);
    const zi = p1.z + s * (p2.z - p1.z);
    path[path.length - 1] = { t, x: xi, y: plateY, z: zi };
    r.set(xi, plateY, zi);
  }

  return { rEnd: r, vEnd: v, t, path };
}
function aimToTarget(r0, v0, omega0, targetXZ, {
  plateY = 0, dt = 1/1000, cdParams = [0.33,0.02], clParams = [0.40,0.35],
  aSSW_body = null, maxIter = 6
} = {}) {
  let yaw = 0, pitch = 0;
  for (let it = 0; it < maxIter; it++) {
    const R = rotationFromYawPitch(yaw, pitch);
    const vT = v0.clone().applyMatrix4(R);
    const oT = omega0.clone().applyMatrix4(R);
    const sim = simulateToPlate(r0, vT, oT, { dt, plateY, cdParams, clParams, aSSW_body, R_body: R, sample: false });
    const ex = sim.rEnd.x - targetXZ.x;
    const ez = sim.rEnd.z - targetXZ.z;
    if (Math.hypot(ex, ez) < (0.05/12)) return { yaw, pitch, sim };

    const h = 1e-3;
    const Ry = rotationFromYawPitch(yaw + h, pitch);
    const Rp = rotationFromYawPitch(yaw, pitch + h);
    const sY = simulateToPlate(r0, v0.clone().applyMatrix4(Ry), omega0.clone().applyMatrix4(Ry),
                               { dt, plateY, cdParams, clParams, aSSW_body, R_body: Ry, sample: false });
    const sP = simulateToPlate(r0, v0.clone().applyMatrix4(Rp), omega0.clone().applyMatrix4(Rp),
                               { dt, plateY, cdParams, clParams, aSSW_body, R_body: Rp, sample: false });

    const dX_dYaw = (sY.rEnd.x - sim.rEnd.x)/h, dZ_dYaw = (sY.rEnd.z - sim.rEnd.z)/h;
    const dX_dPit = (sP.rEnd.x - sim.rEnd.x)/h, dZ_dPit = (sP.rEnd.z - sim.rEnd.z)/h;
    const det = dX_dYaw*dZ_dPit - dZ_dYaw*dX_dPit || 1e-9;
    const dYaw   = (-ex*dZ_dPit +  ez*dX_dPit) / det;
    const dPitch = (-dX_dYaw*ez + dZ_dYaw*ex) / det;

    const LIM = 5*Math.PI/180;
    yaw   = Math.max(-LIM, Math.min(LIM, yaw + dYaw));
    pitch = Math.max(-LIM, Math.min(LIM, pitch + dPitch));
  }
  const R = rotationFromYawPitch(yaw, pitch);
  const sim = simulateToPlate(r0, v0.clone().applyMatrix4(R), omega0.clone().applyMatrix4(R),
                              { dt: 1/1000, plateY, cdParams, clParams, aSSW_body, R_body: R, sample: false });
  return { yaw, pitch, sim };
}
function buildOmega(row, v0) {
  if (row.release_spin_rate && row.spin_axis !== undefined && row.spin_axis !== null) {
    const w_mag = rpm2radS(row.release_spin_rate);
    const theta = (row.spin_axis) * Math.PI / 180;
    const x = Math.sin(theta), z = Math.cos(theta);
    return new THREE.Vector3(x, 0, z).normalize().multiplyScalar(w_mag);
  }
  const axis = estimateSpinAxisFromAccel(row.ax, row.az, v0);
  return axis.multiplyScalar(0); // magnitude unknown unless inferred elsewhere
}
function timeToPlate(row, plateY = 0) {
  const y0 = row.release_pos_y - plateY;
  const a = 0.5 * row.ay;
  const b = row.vy0;
  const c = y0;
  const disc = b*b - 4*a*c;
  if (disc < 0) return null;
  const t1 = (-b - Math.sqrt(disc)) / (2*a);
  const t2 = (-b + Math.sqrt(disc)) / (2*a);
  const t = [t1, t2].filter(v => v > 0).sort()[0];
  return t ?? null;
}
function isStrikeAt(row, top, bot) {
  const half = 17 / 24;
  const inW = Math.abs(row.plate_x) <= half;
  const inH = row.plate_z >= bot && row.plate_z <= top;
  return inW && inH;
}
function buildPathODE(row, {
  dt = 1/1000, plateY = 0,
  cdParams = [0.33, 0.02], clParams = [0.40, 0.35],
  aSSW_body = null, targetXZ = null
} = {}) {
  const r0 = new THREE.Vector3(row.release_pos_x, row.release_pos_y, row.release_pos_z);
  const v0 = new THREE.Vector3(row.vx0, row.vy0, row.vz0);
  const om = buildOmega(row, v0.clone());

  let sim;
  if (targetXZ) {
    const shoot = aimToTarget(r0, v0, om, targetXZ, { plateY, dt, cdParams, clParams, aSSW_body });
    const R     = rotationFromYawPitch(shoot.yaw, shoot.pitch);
    const vT    = v0.clone().applyMatrix4(R);
    const oT    = om.clone().applyMatrix4(R);
    sim = simulateToPlate(r0, vT, oT, { plateY, dt, cdParams, clParams, aSSW_body, R_body: R, sample: true });
  } else {
    sim = simulateToPlate(r0, v0, om, { plateY, dt, cdParams, clParams, aSSW_body, R_body: new THREE.Matrix4(), sample: true });
  }

  const pts = sim.path.map(p => new THREE.Vector3(p.x, p.z, -p.y));
  return { pts, dt, plate_x_sim: sim.rEnd.x, plate_z_sim: sim.rEnd.z };
}

// ───────────────────────── UI helpers ─────────────────────────
function TargetPad({ zoneTop, zoneBot, marginX=1.125, marginZ=0.5, tx, tz, onPick }) {
  const W = 280, H = 360;
  const xMin = -marginX, xMax = +marginX;
  const zMin = zoneBot - marginZ, zMax = zoneTop + marginZ;

  const toScreenX = x => ((x - xMin) / (xMax - xMin)) * W;
  const toScreenY = z => H - ((z - zMin) / (zMax - zMin)) * H;

  const onClick = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const x = xMin + (px / rect.width)  * (xMax - xMin);
    const z = zMin + ((rect.height - py) / rect.height) * (zMax - zMin);
    onPick?.(
      Math.max(xMin, Math.min(xMax, x)),
      Math.max(zMin, Math.min(zMax, z))
    );
  };

  const zoneY1 = toScreenY(zoneTop);
  const zoneY2 = toScreenY(zoneBot);
  const zoneX1 = toScreenX(-17/24);
  const zoneX2 = toScreenX(+17/24);
  const dotX = toScreenX(tx);
  const dotY = toScreenY(tz);

  return (
    <svg width={W} height={H}
      style={{ display:"block", margin:"10px auto", background:"#0e0e0e", border:"1px solid #333", borderRadius:8, cursor:"crosshair" }}
      onClick={onClick}>
      <rect x={0} y={0} width={W} height={H} fill="#0e0e0e" stroke="#222" />
      <rect x={zoneX1} y={zoneY1} width={zoneX2-zoneX1} height={zoneY2-zoneY1}
        fill="transparent" stroke="#00aaff" strokeWidth="2"/>
      {[...Array(8)].map((_,i)=>(<line key={"v"+i} x1={((i+1)/9)*W} y1={0} x2={((i+1)/9)*W} y2={H} stroke="#222" />))}
      {[...Array(8)].map((_,i)=>(<line key={"h"+i} x1={0} y1={((i+1)/9)*H} x2={W} y2={((i+1)/9)*H} stroke="#222" />))}
      <circle cx={dotX} cy={dotY} r={6} fill="#ff7755" stroke="#111"/>
    </svg>
  );
}

// crude residual fit from many rows of same pitch type
function fitResidual(rows, opts) {
  let dxSum = 0, dzSum = 0, n = 0;
  for (const row of rows) {
    const { plate_x_sim, plate_z_sim } = buildPathODE(row, opts);
    dxSum += (row.plate_x - plate_x_sim);
    dzSum += (row.plate_z - plate_z_sim);
    n++;
  }
  if (n === 0) return null;
  const tof = timeToPlate(rows[0], 0) ?? 0.45;
  const axBias = (2 * (dxSum/n)) / (tof*tof);
  const azBias = (2 * (dzSum/n)) / (tof*tof);
  return new THREE.Vector3(axBias, 0, azBias);
}

export default function App() {
  // UI state
  const [id, setId] = useState("");
  const [rows, setRows] = useState([]);          // per-type averages
  const [sel, setSel] = useState(null);
  const [speed, setSpeed] = useState(1);
  const [showTrail, setShowTrail] = useState(true);
  const [zoneTop, setZoneTop] = useState(3.5);
  const [zoneBot, setZoneBot] = useState(1.5);
  const [showTargeter, setShowTargeter] = useState(false);
  const [targeting, setTargeting] = useState(false);
  const [tx, setTx] = useState(0);
  const [tz, setTz] = useState(2.5);
  const [cam, setCam] = useState("catcher");
  const [loadingNearest, setLoadingNearest] = useState(false);
  const [nearestInfo, setNearestInfo] = useState(null);
  const [landX, setLandX] = useState(null);
  const [landZ, setLandZ] = useState(null);

  // raw rows + residuals
  const [allRows, setAllRows] = useState([]);
  const [residualByType, setResidualByType] = useState({});

  // refs to share with three.js closures
  const mountRef = useRef(null);
  const anim = useRef({});
  const targetRef = useRef({ targeting: false, tx: 0, tz: 2.5 });
  const residualRef = useRef({});
  const idRef = useRef(""); 
  

  useEffect(() => { idRef.current = id; }, [id]);  

  useEffect(() => {
    targetRef.current = { targeting, tx, tz };
  }, [targeting, tx, tz]);
  useEffect(() => {
    residualRef.current = residualByType;
  }, [residualByType]);

  useEffect(() => {
    const mount = mountRef.current;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x101010);

    // ── Camera
    const camera = new THREE.PerspectiveCamera(
      52,
      mount.clientWidth / mount.clientHeight,
      0.1,
      1000
    );
    camera.position.set(0, 4, 8);
    camera.lookAt(0, 3, -55);

    anim.current.setCamera = (preset) => {
      switch (preset) {
        case "catcher": camera.position.set(0, 4, 8);      camera.lookAt(0, 3, -55); break;
        case "rhb":     camera.position.set(-2.5, 5.8, 2.0);camera.lookAt(0, 3, -55); break;
        case "lhb":     camera.position.set( 2.5, 5.8, 2.0);camera.lookAt(0, 3, -55); break;
        case "tv":      camera.position.set(0, 10, -80);    camera.lookAt(0, 3.25, 0.25); break;
      }
      if (anim.current.controls) {
        anim.current.controls.target.set(0, 3, -55);
        anim.current.controls.update();
      }
    };
    anim.current.setCamera(cam);

    // ── Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    const W = mount.clientWidth || window.innerWidth;
    const H = mount.clientHeight || window.innerHeight;
    renderer.setSize(W, H);
    mount.appendChild(renderer.domElement);

    // ── Lights + controls
    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dir = new THREE.DirectionalLight(0xffffff, 0.8);
    dir.position.set(5, 10, 5);
    scene.add(dir);

    const controls = new OrbitControls(camera, renderer.domElement);
    if (anim.current.setCamera) anim.current.setCamera(cam);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, 3, 0);
    controls.update();
    controls.minDistance = 5;
    controls.maxDistance = 80;
    controls.minPolarAngle = 0.05;
    controls.maxPolarAngle = Math.PI * 0.51;
    anim.current.controls = controls;

    // ── “Cloud” of neighbor endpoints (replaces search circle)
    let cloudPoints = null;
    function clearCloud() {
      if (cloudPoints) {
        scene.remove(cloudPoints);
        cloudPoints.geometry?.dispose();
        cloudPoints.material?.dispose();
        cloudPoints = null;
      }
    }
    function drawCloud(points) {
      clearCloud();
      if (!points?.length) return;

      const n = points.length;
      const positions = new Float32Array(n * 3);
      const colors    = new Float32Array(n * 3);

      // normalize weights to [0,1] for opacity/brightness
      let maxW = 0;
      for (const p of points) maxW = Math.max(maxW, p.w ?? 0);
      const eps = 1e-6;

      for (let i = 0; i < n; i++) {
        const p = points[i];
        positions[3*i+0] = p.x;  // plate x
        positions[3*i+1] = p.z;  // plate z
        positions[3*i+2] = 0;    // plate plane

        const wNorm = Math.min(1, (p.w ?? 0) / (maxW + eps));
        const c = 0.4 + 0.6 * wNorm; // grayscale 0.4–1.0
        colors[3*i+0] = c;
        colors[3*i+1] = c;
        colors[3*i+2] = c;
      }

      const geom = new THREE.BufferGeometry();
      geom.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
      geom.setAttribute("color",    new THREE.Float32BufferAttribute(colors, 3));

      // single size; PointsMaterial doesn’t do per-vertex size without shader
      const avgSize = Math.max(4, Math.min(12, Math.round(6 + Math.log(points.length))));
      const mat = new THREE.PointsMaterial({
        size: avgSize / 100,     // world units; tweak to taste
        vertexColors: true,
        transparent: true,
        opacity: 0.7,
        depthWrite: false,
        sizeAttenuation: true,
      });

      cloudPoints = new THREE.Points(geom, mat);
      scene.add(cloudPoints);
    }

    // ── Field + visuals
    const plate = new THREE.Mesh(
      new THREE.BoxGeometry(1.417, 0.05, 0.5),
      new THREE.MeshBasicMaterial({ color: 0xffffff })
    );
    plate.position.set(0, 0.025, 0);
    scene.add(plate);

    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(120, 120),
      new THREE.MeshStandardMaterial({ color: 0x0e0e0e, roughness: 1 })
    );
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = 0;
    scene.add(ground);

    const moundY = 60.5, moundZ = -moundY, moundHeight = 0.833;
    const mound = new THREE.Mesh(
      new THREE.CylinderGeometry(2.0, 6.0, moundHeight, 32, 1),
      new THREE.MeshStandardMaterial({ color: 0x8b5a2b, roughness: 0.9 })
    );
    mound.position.set(0, moundHeight / 2, moundZ);
    scene.add(mound);

    const PLATE_HALF = 17 / 24;
    const BOX_W = 4.0, BOX_L = 6.0, BOX_H = 0.02;
    const boxX = PLATE_HALF + 0.5 + BOX_W / 2;
    const boxZ = 0.5 + BOX_L / 2;
    const boxMat = new THREE.MeshStandardMaterial({ color: 0x404040, roughness: 1, metalness: 0 });
    const rhbBox = new THREE.Mesh(new THREE.BoxGeometry(BOX_W, BOX_H, BOX_L), boxMat);
    rhbBox.position.set(+boxX, BOX_H / 2, +boxZ);
    scene.add(rhbBox);
    const lhbBox = new THREE.Mesh(new THREE.BoxGeometry(BOX_W, BOX_H, BOX_L), boxMat);
    lhbBox.position.set(-boxX, BOX_H / 2, +boxZ);
    scene.add(lhbBox);

    const BASE_OFF = 90 / Math.SQRT2, FOUL_LEN = 300, FOUL_W = 0.08, FOUL_H = 0.02;
    const foulMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.9 });
    function addFoulLine(dirVec3) {
      const geom = new THREE.BoxGeometry(FOUL_W, FOUL_H, FOUL_LEN);
      const mesh = new THREE.Mesh(geom, foulMat);
      mesh.position.y = FOUL_H / 2;
      const tgt = dirVec3.clone().normalize();
      const m = new THREE.Matrix4();
      const q = new THREE.Quaternion();
      const up = new THREE.Vector3(0, 1, 0);
      const zAxis = tgt.clone();
      const xAxis = new THREE.Vector3().crossVectors(up, zAxis).normalize();
      const yAxis = new THREE.Vector3().crossVectors(zAxis, xAxis).normalize();
      m.makeBasis(xAxis, yAxis, zAxis);
      q.setFromRotationMatrix(m);
      mesh.quaternion.copy(q);
      mesh.position.add(tgt.multiplyScalar(FOUL_LEN / 2));
      scene.add(mesh);
    }
    addFoulLine(new THREE.Vector3(+BASE_OFF, 0, -BASE_OFF));
    addFoulLine(new THREE.Vector3(-BASE_OFF, 0, -BASE_OFF));

    const field = new THREE.Group();
    scene.add(field);
    const grass = new THREE.Mesh(
      new THREE.PlaneGeometry(400, 400),
      new THREE.MeshStandardMaterial({ color: 0x2d8f2d, roughness: 1, metalness: 0 })
    );
    grass.rotation.x = -Math.PI / 2;
    grass.position.y = 0;
    field.add(grass);

    const infieldDirt = new THREE.Mesh(
      new THREE.CircleGeometry(95, 64),
      new THREE.MeshStandardMaterial({ color: 0x8b5a2b, roughness: 1 })
    );
    infieldDirt.rotation.x = -Math.PI / 2;
    infieldDirt.position.y = 0.005;
    field.add(infieldDirt);

    const BASE_SIZE = 15 / 12, baseH = 0.03;
    const baseGeo = new THREE.PlaneGeometry(BASE_SIZE, BASE_SIZE);
    const baseMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.8, side: THREE.DoubleSide });
    function placeBase(x, z) {
      const p = new THREE.Mesh(baseGeo, baseMat);
      p.rotation.x = -Math.PI / 2;
      p.position.set(x, baseH / 2, z);
      scene.add(p);
    }
    placeBase(+BASE_OFF, -BASE_OFF);
    placeBase(0, -2 * BASE_OFF);
    placeBase(-BASE_OFF, -BASE_OFF);

    // ── strike zone box
    const zone = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(1.417, 2.0, 0.05)),
      new THREE.LineBasicMaterial({ color: 0x00aaff })
    );
    zone.position.set(0, 3.25, 0.25);
    scene.add(zone);

    // ── ball + trail
    const ball = new THREE.Mesh(
      new THREE.SphereGeometry(0.12, 16, 16),
      new THREE.MeshStandardMaterial({ color: 0xffddaa })
    );
    scene.add(ball);

    const trail = new THREE.Line(
      new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({ color: 0xff4444 })
    );
    scene.add(trail);
    trail.visible = showTrail;

    // ── plate marker + residual line (keep residual line if you still draw it)
    const plateMarker = new THREE.Mesh(
      new THREE.SphereGeometry(0.07, 8, 8),
      new THREE.MeshBasicMaterial({ color: 0x00ff00 })
    );
    scene.add(plateMarker);

    let residLine = null;
    const residMat = new THREE.LineBasicMaterial({ color: 0x00ff00 });
    function drawResidual(simX, simZ, plateX, plateZ) {
      if (residLine) {
        scene.remove(residLine);
        residLine.geometry.dispose();
      }
      const geom = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(simX,  simZ, 0),
        new THREE.Vector3(plateX, plateZ, 0),
      ]);
      residLine = new THREE.Line(geom, residMat);
      scene.add(residLine);
    }

    // ── animation state
    const clock = new THREE.Clock();
    let path = [];
    let dtPath = 1 / 480;
    let cursor = 0;
    let playing = true;

    // ── nearest fetch (uses idRef)
    async function fetchNearestRow(mlbamId, pitchType, tx, tz) {
      if (!mlbamId) throw new Error("No pitcher selected");
      const url = new URL(`${API}/pitcher/${mlbamId}/nearest`);
      url.searchParams.set("pitch_type", pitchType);
      url.searchParams.set("x", String(tx));
      url.searchParams.set("z", String(tz));
      url.searchParams.set("k", "120");
      url.searchParams.set("radius_in", "6");
      url.searchParams.set("expand_step_in", "3");
      url.searchParams.set("max_radius_in", "24");
      url.searchParams.set("start", SEASON_START);
      url.searchParams.set("end", SEASON_END);

      const r = await fetch(url.toString());
      if (!r.ok) {
        let msg = `Nearest fetch failed (${r.status})`;
        try {
          const j = await r.json();
          if (j?.detail) msg += ` – ${j.detail}`;
        } catch {}
        throw new Error(msg);
      }
      return await r.json();
    }

    // ── loadPitch: does the cloud + path integration
    async function loadPitch(row) {
      if (!row) return;

      const { targeting: tg, tx: TX, tz: TZ } = targetRef.current;
      let odeRow = row;

      try {
        if (tg) {
          setLoadingNearest(true);
          const agg = await fetchNearestRow(idRef.current, row.pitch_type, TX, TZ);
          odeRow = agg;

          setNearestInfo({
            neighbors_used: agg.neighbors_used,
            mean_dist_in:   agg.mean_dist_in,
            search_radius_in: agg.search_radius_in,
          });

          // DRAW THE CLOUD (replaces the old search circle)
          if (Array.isArray(agg.points)) drawCloud(agg.points);
          else clearCloud();
        } else {
          setNearestInfo(null);
          clearCloud();
        }
      } catch (e) {
        console.warn("Nearest lookup failed, using averaged row:", e);
        setNearestInfo(null);
        clearCloud();
        odeRow = row;
      } finally {
        setLoadingNearest(false);
      }

      // color by pitch type
      const col = TYPE_COLOR[odeRow.pitch_type] ?? 0xff4444;
      ball.material.color.setHex(col);
      trail.material.color.setHex(col);

      // strike zone height/pos from UI
      const top = zoneTop;
      const bot = zoneBot;
      const h = (top - bot) || 2.0;
      const center = bot + h / 2;
      zone.position.set(0, center, 0.25);
      zone.scale.set(1, h / 2.0, 1);

      // residual bias per pitch type
      const residualByTypeNow = residualRef.current || {};
      const bias = residualByTypeNow[odeRow.pitch_type] || null;

      // integrate trajectory
      const { pts, dt } = buildPathODE(odeRow, {
        dt: 1/1000,
        plateY: 0,
        cdParams: [0.33, 0.02],
        clParams: [0.40, 0.35],
        aSSW_body: bias,
        targetXZ: null,
      });

      // update trail geometry safely (dispose old, swap new)
      const positions = new Float32Array(pts.length * 3);
      for (let i = 0; i < pts.length; i++) {
        positions[3*i+0] = pts[i].x;
        positions[3*i+1] = pts[i].y;
        positions[3*i+2] = pts[i].z;
      }
      const newGeom = new THREE.BufferGeometry();
      newGeom.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
      newGeom.computeBoundingSphere();
      trail.geometry?.dispose();
      trail.geometry = newGeom;
      trail.geometry.setDrawRange(0, pts.length);
      trail.visible = showTrail;

      // reset animation
      path = pts;
      dtPath = dt;
      cursor = 0;
      clock.elapsedTime = 0;
      clock.getDelta();
      playing = true;

      // move the green marker to actual terminal location
      plateMarker.position.set(odeRow.plate_x ?? 0, odeRow.plate_z ?? 0, 0);
      setLandX(odeRow.plate_x ?? null);
      setLandZ(odeRow.plate_z ?? null);
      // (optional) drawResidual(...) if you still want the vector
      // drawResidual(simX, simZ, odeRow.plate_x ?? 0, odeRow.plate_z ?? 0);
    }
    anim.current.load = loadPitch;

    // ── tick
    function animate() {
      anim.current.raf = requestAnimationFrame(animate);
      const delta = clock.getDelta();

      if (path.length && playing) {
        const advance = (delta / dtPath) * speed;
        cursor = Math.min(cursor + advance, path.length - 1);
        const i = Math.floor(cursor);
        ball.position.copy(path[i]);

        if (showTrail && trail.geometry) {
          trail.geometry.setDrawRange(0, i + 1);
          trail.visible = true;
        } else {
          trail.visible = false;
        }
      }

      controls.update();
      renderer.render(scene, camera);
    }
    animate();

    // ── resize & cleanup
    function onResize() {
      camera.aspect = mount.clientWidth / mount.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mount.clientWidth, mount.clientHeight);
    }
    window.addEventListener("resize", onResize);

    return () => {
      cancelAnimationFrame(anim.current.raf);
      window.removeEventListener("resize", onResize);
      clearCloud();
      mount.removeChild(renderer.domElement);
      renderer.dispose();
      controls.dispose();
    };
  }, [speed, showTrail, zoneTop, zoneBot, cam]);


  useEffect(() => {
    if (anim.current.setCamera) anim.current.setCamera(cam);
  }, [cam]);

useEffect(() => {
  if (!sel || !anim.current.load) return;
  setLandX(null);
  setLandZ(null);
  (async () => { await anim.current.load(sel); })();
}, [sel, targeting, tx, tz, residualByType]);



  // ------------------- data fetching -------------------
  async function fetchAveragesAndRows() {
    if (!id) return;

    // include start/end in case your backend now requires them
    const avgUrl  = new URL(`${API}/pitcher/${id}/averages`);
    avgUrl.searchParams.set("start", SEASON_START);
    avgUrl.searchParams.set("end", SEASON_END);

    const rowsUrl = new URL(`${API}/pitcher/${id}/rows`);
    rowsUrl.searchParams.set("start", SEASON_START);
    rowsUrl.searchParams.set("end", SEASON_END);

    let avgJson = null, rowsJson = null;

    try {
      const [avgRes, rowsRes] = await Promise.all([ fetch(avgUrl), fetch(rowsUrl) ]);
      avgJson  = await safeJson(avgRes);
      rowsJson = await safeJson(rowsRes);

      // Normalize to arrays or fail-safe empties
      const avgArr  = Array.isArray(avgJson)  ? avgJson  : [];
      const rowsArr = Array.isArray(rowsJson) ? rowsJson : [];

      if (!avgRes.ok || !Array.isArray(avgJson)) {
        console.warn("Averages fetch failed:", avgRes.status, avgJson);
      }
      if (!rowsRes.ok || !Array.isArray(rowsJson)) {
        console.warn("Rows fetch failed:", rowsRes.status, rowsJson);
      }

      setRows(avgArr);
      setSel(avgArr[0] || null);
      setAllRows(rowsArr);

      // fit residuals per pitch type if we have rows
      if (rowsArr.length) {
        const byType = {};
        for (const r of rowsArr) (byType[r.pitch_type] ??= []).push(r);
        const baseOpts = { dt: 1/1000, plateY: 0, cdParams:[0.33,0.02], clParams:[0.40,0.35] };
        const fitted = {};
        for (const [pt, group] of Object.entries(byType)) {
          if (group.length >= 8) fitted[pt] = fitResidual(group, baseOpts);
        }
        setResidualByType(fitted);
      } else {
        setResidualByType({});
      }
    } catch (e) {
      console.error("fetchAveragesAndRows exception:", e);
      // fail-safe: clear lists so render doesn’t break
      setRows([]);
      setSel(null);
      setAllRows([]);
      setResidualByType({});
    }
  }


  // ------------------- render -------------------
  return (
    <div style={{ display: "grid", gridTemplateColumns: "320px 1fr", height: "100vh" }}>
      <div style={{ padding: 16, background: "#0b0b0b", color: "#eee", overflow: "auto" }}>
        <h2>Pitch3D</h2>

        <div style={{ marginBottom: 12 }}>
          <label>MLBAM ID:&nbsp;</label>
          <input value={id} onChange={e => setId(e.target.value)} placeholder="e.g. 543037 (Gerrit Cole)" />
          <button onClick={fetchAveragesAndRows} style={{ marginLeft: 8 }}>Load</button>
        </div>

        <div style={{ margin: "12px 0" }}>
          <div>Speed: {speed.toFixed(2)}x</div>
          <input type="range" min="0.25" max="2" step="0.05" value={speed}
            onChange={e => setSpeed(parseFloat(e.target.value))}/>
        </div>

        <div style={{ marginTop: 8 }}>
          <strong>Camera</strong>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4,auto)", gap: 6, marginTop: 6 }}>
            <button onClick={() => setCam("catcher")} style={{ opacity: cam==="catcher"?1:0.7 }}>Catcher</button>
            <button onClick={() => setCam("rhb")}     style={{ opacity: cam==="rhb"?1:0.7 }}>RHB POV</button>
            <button onClick={() => setCam("lhb")}     style={{ opacity: cam==="lhb"?1:0.7 }}>LHB POV</button>
            <button onClick={() => setCam("tv")}      style={{ opacity: cam==="tv"?1:0.7 }}>Behind Mound</button>
          </div>
        </div>

        <div style={{ margin: "8px 0" }}>
          <label>
            <input type="checkbox" checked={showTrail} onChange={e => setShowTrail(e.target.checked)} />
            &nbsp;Show tracer
          </label>
        </div>

        <div style={{ marginTop: 8, display: "flex", gap: 8, alignItems: "center" }}>
          <button onClick={() => setShowTargeter(true)} disabled={!id || !sel}>Aim Pitch</button>
          <button
            onClick={() => {
              setTargeting(false);
              setLandX(null);
              setLandZ(null);
              if (sel && anim.current.load) anim.current.load(sel);
            }}
            disabled={!targeting}
          >
            Reset Aim
          </button>
        </div>

        {loadingNearest && (
          <div style={{opacity:0.8, fontSize:12}}>Finding nearest examples…</div>
        )}

        {targeting && nearestInfo && (
          <div style={{ marginTop: 6, fontSize: 12, opacity: 0.75 }}>
            {nearestInfo.neighbors_used ?? "—"} comps ·
            {" "}mean {nearestInfo.mean_dist_in?.toFixed?.(1) ?? "—"}″ ·
            {" "}radius {nearestInfo.search_radius_in?.toFixed?.(0) ?? "—"}″
          </div>
        )}

        <hr />

        {sel && (() => {
          const name = TYPE_NAME[sel.pitch_type] || sel.pitch_type;
          const tof = timeToPlate(sel);
          const tofMs = tof ? `${(tof*1000).toFixed(0)} ms` : "—";
          const top = zoneTop;
          const bot = zoneBot;
          const px = landX ?? sel.plate_x;
          const pz = landZ ?? sel.plate_z;
          const zoneText = (Math.abs(px) <= (17/24) && pz >= bot && pz <= top) ? "In Zone" : "Out of Zone";
          return (
            <div style={{
              margin: "8px 0 12px",
              padding: 10,
              background: "#151515",
              border: "1px solid #333",
              borderRadius: 8
            }}>
              <div style={{ fontWeight: 700, fontSize: 18 }}>
                {name} <span style={{ opacity: 0.7 }}>({sel.pitch_type})</span>
              </div>
              <div style={{ marginTop: 4, opacity: 0.9 }}>
                {sel.release_speed?.toFixed(1)} mph · TOF {tofMs}
              </div>
              <div style={{ marginTop: 2, fontSize: 13, opacity: 0.8 }}>
                Landed x {px?.toFixed(2)} ft · z {pz?.toFixed(2)} ft · <strong>{zoneText}</strong>
              {targeting && (
                <span style={{ opacity: 0.7 }}>
                  {" "} (aimed x {tx.toFixed(2)}, z {tz.toFixed(2)})
                </span>
                )}
              </div>
              <div style={{ marginTop: 2, fontSize: 12, opacity: 0.7 }}>
                Usage: {sel.usage_pct?.toFixed(1)}%
              </div>
            </div>
          );
        })()}

        {showTargeter && (
          <div
            style={{
              position: "fixed", inset: 0, background: "rgba(0,0,0,0.6)",
              display: "grid", placeItems: "center", zIndex: 9999
            }}
            onClick={() => setShowTargeter(false)}
          >
            <div
              style={{ width: 320, padding: 12, background: "#151515", border: "1px solid #333", borderRadius: 10 }}
              onClick={e => e.stopPropagation()}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <strong>Target Pitch</strong>
                <button onClick={() => setShowTargeter(false)}>✕</button>
              </div>

              <TargetPad
                zoneTop={zoneTop} zoneBot={zoneBot}
                marginX={1.125} marginZ={0.5}
                tx={tx} tz={tz}
                onPick={(x, z) => {
                  setTx(x);
                  setTz(z);
                  setTargeting(true);
                  setShowTargeter(false);
                  if (sel && anim.current.load) anim.current.load(sel);
                }}
              />

              <div style={{ marginTop: 8, fontSize: 12, opacity: 0.8 }}>
                Click anywhere to set terminal location.
                X: {tx.toFixed(2)} ft, Z: {tz.toFixed(2)} ft
              </div>
            </div>
          </div>
        )}

        <strong>Pitch Types</strong>
        <ul style={{ listStyle: "none", paddingLeft: 0 }}>
          {(Array.isArray(rows) ? rows : []).map(r => (
            <li key={r.pitch_type} style={{ margin: "6px 0" }}>
              <button
                onClick={() => setSel(r)}
                style={{
                  background: sel?.pitch_type === r.pitch_type ? "#2a2a2a" : "#1a1a1a",
                  borderColor: sel?.pitch_type === r.pitch_type ? "#646cff" : "transparent"
                }}
              >
                {(TYPE_NAME[r.pitch_type] || r.pitch_type)} — {r.usage_pct?.toFixed(1)}%
              </button>
            </li>
          ))}
        </ul>
        <p style={{ fontSize: 12, opacity: 0.7 }}>Data: MLB Statcast / Baseball Savant</p>
      </div>

      <div ref={mountRef} style={{ width: "100%", height: "100%", background: "#111" }} />
    </div>
  );
}
