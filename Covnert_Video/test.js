const { Pose, Hand } = require("kalidokit");
const fs = require("fs");

// 데이터 불러오기
const pose3d = JSON.parse(fs.readFileSync('pose3d.json', 'utf8')); // [frame][33]
const handL3d = JSON.parse(fs.readFileSync('hand_left3d.json', 'utf8')); // [frame][21]
const handR3d = JSON.parse(fs.readFileSync('hand_right3d.json', 'utf8')); // [frame][21]

let output = [];

for (let i = 0; i < pose3d.length; i++) {
  const poseLandmarks = pose3d[i];
  const leftHandLandmarks = handL3d[i];
  const rightHandLandmarks = handR3d[i];

  // Kalidokit 변환
  const poseRig = Kalidokit.Pose.solve(poseLandmarks, null, {runtime: 'mediapipe' , enableLegs: false});
  const leftHandRig = Kalidokit.Hand.solve(leftHandLandmarks, "Left");
  const rightHandRig = Kalidokit.Hand.solve(rightHandLandmarks, "Right");


  output.push([
    {"pose": poseRig},      // 신체 본 회전값
    {"hand_left": leftHandRig},   // 왼손가락 본 회전값
    {"hand_right": rightHandRig}  // 오른손가락 본 회전값
  ]);
}

fs.writeFileSync('holistic_rigged_output.json', JSON.stringify(output, null, 2));
console.log('변환 완료: holistic_rigged_output.json');
