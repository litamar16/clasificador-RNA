let modelo;

// ==========================================
// COMPROBAR SI UN NÚMERO ES PRIMO
// ==========================================

function esPrimo(numero) {

  if (numero < 2) {
    return false;
  }

  for (let i = 2; i <= Math.sqrt(numero); i++) {

    if (numero % i === 0) {
      return false;
    }

  }

  return true;
}


// ==========================================
// ENTRENAR RED NEURONAL
// ==========================================

async function entrenarRNA() {

  document.getElementById("estado").innerText =
    "Entrenando RNA...";

  const entradas = [];
  const salidas = [];


  // Datos de entrenamiento
  // 0 = PAR
  // 1 = IMPAR

  for (let numero = 0; numero <= 100; numero++) {

    entradas.push([numero / 100]);

    if (numero % 2 === 0) {
      salidas.push([0]);
    } else {
      salidas.push([1]);
    }

  }


  // Convertir datos a tensores

  const xs = tf.tensor2d(entradas);
  const ys = tf.tensor2d(salidas);


  // Crear RNA

  modelo = tf.sequential();


  // Primera capa

  modelo.add(
    tf.layers.dense({
      inputShape: [1],
      units: 16,
      activation: "relu"
    })
  );


  // Segunda capa

  modelo.add(
    tf.layers.dense({
      units: 8,
      activation: "relu"
    })
  );


  // Capa de salida

  modelo.add(
    tf.layers.dense({
      units: 1,
      activation: "sigmoid"
    })
  );


  // Configurar RNA

  modelo.compile({

    optimizer: tf.train.adam(0.01),

    loss: "binaryCrossentropy",

    metrics: ["accuracy"]

  });


  // Entrenar

  await modelo.fit(xs, ys, {

    epochs: 100,

    shuffle: true,

    callbacks: {

      onEpochEnd: (epoca) => {

        document.getElementById("estado").innerText =
          "Entrenando época " + (epoca + 1) + "/100";

      }

    }

  });


  // Liberar memoria

  xs.dispose();
  ys.dispose();


  document.getElementById("estado").innerText =
    "RNA entrenada ✅";


  document.getElementById("resultado").innerHTML =
    `
    <h2>🧠 RNA lista</h2>

    <p>
      La Red Neuronal Artificial está preparada
      para analizar números.
    </p>

    <p>
      Introduce un número y pulsa
      <strong>ANALIZAR</strong>.
    </p>
    `;
}


// ==========================================
// ANALIZAR NÚMERO
// ==========================================

async function analizarNumero() {


  // Comprobar que la RNA esté lista

  if (!modelo) {

    alert(
      "La RNA todavía está entrenándose. Espera unos segundos."
    );

    return;

  }


  // Obtener número

  const numero = Number(
    document.getElementById("numero").value
  );


  // Comprobar entrada

  if (
    document.getElementById("numero").value === ""
    || !Number.isInteger(numero)
  ) {

    document.getElementById("resultado").innerHTML =
      "⚠️ Introduce un número entero.";

    return;

  }


  // ==========================================
  // PREDICCIÓN DE LA RNA
  // ==========================================

  const entrada = tf.tensor2d([
    [numero / 100]
  ]);


  const prediccion = modelo.predict(entrada);


  const valor =
    (await prediccion.data())[0];


  entrada.dispose();
  prediccion.dispose();


  // ==========================================
  // PAR / IMPAR SEGÚN LA RNA
  // ==========================================

  let resultadoRNA;
  let confianzaRNA;


  if (valor >= 0.5) {

    resultadoRNA = "IMPAR";

    confianzaRNA = valor * 100;

  } else {

    resultadoRNA = "PAR";

    confianzaRNA = (1 - valor) * 100;

  }


  // ==========================================
  // COMPROBACIÓN MATEMÁTICA
  // ==========================================

  const resultadoMatematico =
    numero % 2 === 0
      ? "PAR"
      : "IMPAR";


  // ==========================================
  // PRIMO / NO PRIMO
  // ==========================================

  const primo = esPrimo(numero);


  const resultadoPrimo =
    primo
      ? "SÍ 🟢"
      : "NO 🔴";


  // ==========================================
  // MOSTRAR RESULTADO
  // ==========================================

  document.getElementById("resultado").innerHTML = `

    <h2>🔢 Número: ${numero}</h2>

    <hr>

    <h3>🧠 Red Neuronal Artificial</h3>

    <p>
      Predicción:
      <strong>${resultadoRNA}</strong>
    </p>

    <p>
      Confianza:
      <strong>${confianzaRNA.toFixed(2)}%</strong>
    </p>

    <hr>

    <h3>🔬 Análisis matemático</h3>

    <p>
      Paridad:
      <strong>${resultadoMatematico}</strong>
    </p>

    <p>
      ¿Es primo?:
      <strong>${resultadoPrimo}</strong>
    </p>

  `;
}


// ==========================================
// INICIAR ENTRENAMIENTO
// ==========================================

entrenarRNA();
