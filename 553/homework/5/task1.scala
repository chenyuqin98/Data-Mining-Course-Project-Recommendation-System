import scala.io.Source
import java.lang.Math.abs
import scala.collection.mutable.MutableList
import scala.util.control.Breaks._
import java.io._


object task1 {
  def main(arg:Array[String]): Unit={
    val input_filename = arg(0)
    val stream_size = arg(1).toInt
    val num_of_asks: Int = arg(2).toInt
    val output_filename = arg(3)

    class Blackbox {
      private val r1 = scala.util.Random
      def ask(filename: String, num: Int): Array[String] = {
        val input_file_path = filename

        val lines = Source.fromFile(input_file_path).getLines().toArray
        var stream = new Array[String](num)

        for (i <- 0 to num - 1) {
          stream(i) = lines(r1.nextInt(lines.length))
        }
        return stream
      }
    }

    // Start timer
    val start_time = System.nanoTime
    // initialize Blackbox
    val box = new Blackbox
    // keep track of previous users
    var previous_users: Set[String] = Set()
    // filter bit array
    val A = MutableList.fill(69997)(0)
    // generate 16 hash function parameters
    val hash_param_list = generate_hash_param(16)
    // keep track of FPR
    var fpr_string = ""
    for (i:Int <- 0 to num_of_asks - 1){
      val stream_users = box.ask(input_filename, stream_size)
      val predictions = MutableList.fill(stream_size)(0)
      val fp = 0
      val tn = 0
      for (j:Int <- 0 to stream_users.length-1){
        val user_s = stream_users(j)
        val hash_rlt = myhashs(user_s, hash_param_list)
        predictions(j) = check(A, hash_rlt)
        A = construct(A, hash_rlt)
        if (!previous_users.contains(user_s) && predictions(j) == 1) {
            fp += 1
        }
        if (!previous_users.contains(user_s)) {
            tn += 1
        }
        previous_users += user_s
      }
      val fpr = fp / (fp + tn)
      fpr_string = fpr_string + i.toString + "," + fpr.toString + "\n"
    }
    var res:String = "Time,FPR\n"
    res = res + fpr_string
    // write file
    val file = new File(output_filename)
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(res)
    bw.close()

    val duration = (System.nanoTime - start_time) / 1e9d
    println("Duration: "+duration)
  }

  def generate_hash_param(function_num: Int): MutableList[List[Int]] = {
    val r = scala.util.Random
    val hash_param_list:MutableList[List[Int]]  = MutableList()
    for (i <- 1 to function_num) {
      var a = abs(r.nextInt(1000))
      var b = abs(r.nextInt(1000))
      hash_set += List(a, b)
    }
    hash_param_list
  }

  def check(A: MutableList[Int], hash_vals:Array[Int]): Int = {
    for (hash_val <- hash_vals) {
        if (A(hash_val) == 0) {
            0
        }
    }
    1
  }

  def construct(A: MutableList[Int], hash_vals:Array[Int]): MutableList[Int] = {
    for (hash_val <- hash_vals) {
        A(hash_val) = 1
    }
    A
  }

  def myhashs(user_string: String, hash_param_list: MutableList[List[Int]]): Array[Int] = {
    var result:Array[Int] = Array()
    val user_num = abs(user_string.hashCode)
    for (hash <- hash_param_list) {
      val hash_val = (hash(0) * user_num + hash(1)) % 69997
      result = result :+ hash_val
    }
    result
  }
}
