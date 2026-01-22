∇ Z←QUERY ⍵
  Z←⎕SQL ⍵
∇

∇ result←alpha CONNECT omega;socket
  socket←⎕SQL alpha omega
  result←socket
∇

∇ Z←∆Process ⍵;⍬temp;⎕IO
  ⎕IO←1
  ⍬temp←⍵+10
  Z←⍬temp×2
∇
