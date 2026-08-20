let modelo;

// Comprobar si es primo
function esPrimo(n) {

  if (n < 2) return false;

  for (let i = 2; i <= Math.sqrt(n); i++) {

    if (n % i === 0) return false;

  }

  return true;
}


// Crear y entrenar RNA
async function entrenarRNA() {

  document.getElementById("estado").innerText =
    "Entrenando RNA...";


  // Datos de entrenamiento
  const entradas = [];
  const salidas = [];


  for (let n = 1; n <= 50; n++) {

    entradas.push([n / 50]);

    // 0 = PAR
    // 1 = IMPAR

    salidas.push([
      n % 2
    ]);

  }


  const xs = tf.tensor2d(entradas);
  const ys = tf.tensor2d(salidas);


  // Crear RNA pequeña

  modelo = tf.sequential();


  modelo.add(
    tf.layers.dense({

      inputShape: [1],

      units: 4,

      activation: "relu"

    })
  );


  modelo.add(
    tf.layers.dense({

      units: 1,

      activation: "sigmoid"

    })
  );


  modelo.compile({

    optimizer: "adam",

    loss: "binaryCrossentropy",

    metrics: ["accuracy"]

  });


  // Entrenar solamente 15 épocas

  await modelo.fit(xs, ys, {

    epochs: 15,

    shuffle: true,

    verbose: 0

  });


  xs.dispose();
  ys.dispose();


  document.getElementById("estado").innerText =
    "RNA lista ✅";


  document.getElementById("resultado").innerHTML = `

    <h2>🧠 RNA preparada</h2>

    <p>
      La Red Neuronal está lista para analizar.
    </p>

  `;

}


// Analizar número
async function analizarNumero() {


  if (!modelo) {

    alert("Espera a que la RNA termine.");

    return;

  }


  const numero =
    Number(
      document.getElementById("numero").value
    );


  if (
    !Number.isInteger(numero) ||
    numero < 1 ||
    numero > 50
  ) {

    alert(
      "Introduce un número entero entre 1 y 50."
    );

    return;

  }


  // RNA

  const entrada =
    tf.tensor2d([
      [numero / 50]
    ]);


  const prediccion =
    modelo.predict(entrada);


  const valor =
    (await prediccion.data())[0];


  entrada.dispose();
  prediccion.dispose();


  let paridad;
  let confianza;


  if (valor >= 0.5) {

    paridad = "IMPAR";

    confianza = valor * 100;

  } else {

    paridad = "PAR";

    confianza = (1 - valor) * 100;

  }


  // Primo

  const primo =
    esPrimo(numero);


  // Mostrar

  document.getElementById("resultado").innerHTML = `

    <div class="numero-grande">
      ${numero}
    </div>


    <div class="resultado-grid">


      <div class="resultado-card">

        <h3>🧠 RNA</h3>

        <p>
          PAR / IMPAR
        </p>

        <div class="prediccion">

          ${paridad}

        </div>

        <p>

          Confianza:
          <strong>
            ${confianza.toFixed(1)}%
          </strong>

        </p>

      </div>


      <div class="resultado-card">

        <h3>🔬 Matemática</h3>

        <p>
          PRIMO / NO PRIMO
        </p>

        <div class="prediccion">

          ${primo
            ? "PRIMO 🟢"
            : "NO PRIMO 🔴"}

        </div>

      </div>


    </div>

  `;

}


// Limpiar
function limpiar() {

  document.getElementById("numero").value = "";

  document.getElementById("resultado")
    .classList.add("oculto");

}


// Entrenar automáticamente
entrenarRNA();
