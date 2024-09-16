extern crate bmp;
use bmp::{Image, Pixel};
use std::time::{Duration, Instant};
use rand::Rng;
use std::sync::{Arc,Mutex,Condvar};
use std::thread;

fn main() {

    //world building
    //let mut particle_system = ParticleSystem::new(0.05, 9.8);
    let mut red_spray_can = SprayCan::new((0.0, 0.2, 0.0),  (1.0, 0.0, 0.0), 10, 45.0_f32.to_radians());
    //let mut blue_spray_can = SprayCan::new((0.1, 0.2, 0.0),  (0.0, 0.0, 1.0), 10, 45.0_f32.to_radians());
    //let mut green_spray_can = SprayCan::new((-0.1, 0.2, 0.0),  (0.0, 1.0, 0.0), 10, 45.0_f32.to_radians());

    let mut paper = Paper::new(1000, 1000, (1.0, 0.1, 1.0), (0.0, -0.1, 0.0));

    //thread handles
    let mut array_of_threads = vec!();

    //particles conditional variable
    let particle_cv = Arc::new((Mutex::new(red_spray_can.particles),Condvar::new()));
    let particles_clone = Arc::clone(&particle_cv);

    //paper conditional variable
    let paper_cv = Arc::new((Mutex::new(paper),Condvar::new()));
    let paper_clone = Arc::clone(&paper_cv);


    //producer
    //create thread - create a number of particles. wait until there are no more particles, create more particles
    array_of_threads.push(thread::spawn(move || {
        let (lock,cv) = &*particles_clone;
        let mut particles = lock.lock().unwrap();
        while !particles.is_empty() {
            particles = cv.wait(particles).unwrap();
        }

        let mut rng = rand::thread_rng();

        for _i in 0..350000 {
            let angle = rng.gen_range(-45.0_f32.to_radians() / 2.0..45.0_f32.to_radians() / 2.0);
            let velocity_x = rng.gen_range(-1.0..1.0); 
            let velocity_z = rng.gen_range(-1.0..1.0); 
    
            let rotated_velocity_x = velocity_x * angle.cos() - velocity_z * angle.sin();
            let rotated_velocity_z = velocity_x * angle.sin() + velocity_z * angle.cos();

            particles.push(Particle::new((0.0,0.2,0.0), (rotated_velocity_x,-1.0,rotated_velocity_z), (1.0,0.0,0.0)))
        }

        cv.notify_all();
    }));

    let particles_clone_2 = Arc::clone(&particle_cv);
    //consumer
    //move thread
    array_of_threads.push(thread::spawn(move || {
        let (lock,cv) = &*particles_clone_2;
        let mut particles = lock.lock().unwrap();
        let (lock2,cv2) = &* paper_clone;
        let mut paper = lock2.lock().unwrap();
        while particles.is_empty() {
            particles = cv.wait(particles).unwrap();
        }

        while !paper.hits.is_empty(){
            paper = cv2.wait(paper).unwrap();
        }
        
        let gravity = 9.8;
        let drag = 0.05;

        let start_time = Instant::now();
        let mut last_frame_time = Instant::now();

        while start_time.elapsed() < Duration::from_secs(2){
            let now = Instant::now();
            let delta_time = now.duration_since(last_frame_time).as_secs_f32();
            last_frame_time = now;

            let mut particles_to_remove = vec![];
            for i in 0..particles.len() {
                
                let velocity_mag = (particles[i].velocity.0.powi(2)
                + particles[i].velocity.1.powi(2)
                + particles[i].velocity.2.powi(2))
                .sqrt();

                let acceleration_mag = gravity - drag * velocity_mag.powi(2);

                let mut normalized_velocity = particles[i].velocity;
                let velocity_mag_inv = 1.0 / velocity_mag;
                normalized_velocity.0 *= velocity_mag_inv;
                normalized_velocity.1 *= velocity_mag_inv;
                normalized_velocity.2 *= velocity_mag_inv;

                let acceleration = (
                    normalized_velocity.0 * acceleration_mag,
                    normalized_velocity.1 * acceleration_mag,
                    normalized_velocity.2 * acceleration_mag,
                );

                particles[i].velocity.0 += acceleration.0 * delta_time;
                particles[i].velocity.1 += acceleration.1 * delta_time;
                particles[i].velocity.2 += acceleration.2 * delta_time;

                particles[i].position.0 += particles[i].velocity.0 * delta_time;
                particles[i].position.1 += particles[i].velocity.1 * delta_time;
                particles[i].position.2 += particles[i].velocity.2 * delta_time;

                if particles[i].position.0 < paper.position.0 + (paper.scale.0 / 2.0)
                    && particles[i].position.0 > paper.position.0 - (paper.scale.0 / 2.0)
                    && particles[i].position.1 < paper.position.1 + (paper.scale.1 / 2.0)
                    && particles[i].position.1 > paper.position.1 - (paper.scale.1 / 2.0)
                    && particles[i].position.2 < paper.position.2 + (paper.scale.2 / 2.0)
                    && particles[i].position.2 > paper.position.2 - (paper.scale.2 / 2.0)
                {
                    paper.hits.push(particles[i].clone());
                    particles_to_remove.push(i); 
                }
                
            }

            for &i in particles_to_remove.iter().rev() {
                particles.remove(i);

            }
        }
        cv.notify_all();
        cv2.notify_all();
    }));
    
    let paper_clone2 = Arc::clone(&paper_cv);

    //calculate the position of the paint particles on the paper
    array_of_threads.push(thread::spawn(move || {
        let(lock,cv) = &*paper_clone2;
        let mut paper = lock.lock().unwrap();

        while paper.hits.is_empty() {
            paper = cv.wait(paper).unwrap();
        }

        let paper_start_x = paper.position.0 - (paper.scale.0 / 2.0); //X(0) position on paper
        let paper_start_y = paper.position.2 - (paper.scale.2 / 2.0); //Y(0) position on paper

        let x_constant = paper.x_capacity as f32 / paper.scale.0;
        let y_constant = paper.y_capacity as f32 / paper.scale.2;

        for i in 0..paper.hits.len() {

            let index_x = ((paper.hits[i].position.0 - paper_start_x) * x_constant) as usize;
            let index_y = ((paper.hits[i].position.2 - paper_start_y) * y_constant) as usize;

            let b = 0.1;
            
            paper.paint_array[index_x][index_y] = ((1.0 - b) * paper.paint_array[index_x][index_y].0 + b * paper.hits[i].colour.0,
                                                (1.0 - b) * paper.paint_array[index_x][index_y].1 + b * paper.hits[i].colour.1,
                                                (1.0 - b) * paper.paint_array[index_x][index_y].2 + b * paper.hits[i].colour.2);

        }

            write_bmp(&paper);
            cv.notify_all();
    }));

    //main loop
    // while start_time.elapsed() < Duration::from_secs(2) {
    //     let now = Instant::now();
    //     let delta_time = now.duration_since(last_frame_time).as_secs_f32();
    //     last_frame_time = now;

    //     red_spray_can.new_spray(&mut particle_system);

    //     particle_system.update(delta_time);
    //     particle_system.CheckCollisionAgainstPaper(&mut paper);

    // }

    for thread in array_of_threads {
        thread.join().unwrap();
    }


}



pub fn write_bmp(paper : &Paper){
    let mut img = Image::new(paper.x_capacity as u32, paper.y_capacity as u32);

    for y in 0..paper.y_capacity  {
        for x in 0..paper.x_capacity {
            let mut pixel = Pixel::new(
                (255.0 * paper.paint_array[x][y].0) as u8,
                (255.0 * paper.paint_array[x][y].1) as u8,
                (255.0 * paper.paint_array[x][y].2) as u8,
            );

            img.set_pixel(x as u32, y as u32, pixel);
        }
    }

    // Save the paper array to a bmp file
    img.save("paper.bmp").unwrap();
    
}

struct Paper {
    x_capacity: usize,
    y_capacity: usize,
    scale: (f32, f32, f32),
    position: (f32, f32, f32),
    paint_array: Vec<Vec<(f32, f32, f32)>>,
    hits : Vec<(Particle)>
}

impl Paper {
    fn new(x_capacity: usize,y_capacity: usize,scale: (f32, f32, f32),position: (f32, f32, f32),) -> Paper {
        let mut paint_array = Vec::with_capacity(x_capacity as usize);
        
        for _ in 0..x_capacity as usize {
            let mut row = Vec::with_capacity(y_capacity as usize);
            for _i in 0..y_capacity as usize {
                row.push((1.0, 1.0, 1.0)); 
            }
            paint_array.push(row);
        }

        Paper {
            x_capacity: x_capacity,
            y_capacity: y_capacity,
            scale: scale,
            position: position,
            paint_array: paint_array,
            hits : vec!()
        }
    }

}

struct SprayCan {
    position: (f32, f32, f32),
    spray_volume: usize,
    colour : (f32,f32,f32), //keep values between 0 and 1
    particles : Vec<Particle>,
    spray_angle : f32
}

impl SprayCan {
    fn new(position: (f32, f32, f32),colour: (f32, f32, f32),spray_volume: usize, spray_angle : f32) -> SprayCan {
        
        SprayCan {
            position: position,
            colour : colour,
            spray_volume: spray_volume,
            spray_angle : spray_angle,
            particles : vec!()
        }
    }

    // fn create_particles(&mut self, size : usize) {
    //     let particle_clone = Arc::clone(&self.particles);
    //     let mut particles = particle_clone.lock().unwrap();
    //     for i in 0..size {
    //         particles.push(Particle::new(self.position, (0.0,0.0,0.0), self.colour));
    //     }
        
    // }

    fn spray(&mut self, particle_system: &mut ParticleSystem){

        let mut rng = rand::thread_rng();

        for _i in 0..self.spray_volume{

    
            let angle = rng.gen_range(0.0..self.spray_angle);
            let velocity_x = rng.gen_range(-angle.cos()..angle.cos());
            let velocity_z = rng.gen_range(-velocity_x.cos()..velocity_x.cos());


            //let velocity_x = rng.gen_range(-0.75..0.75);
            //let velocity_z = rng.gen_range(-0.75..0.75);
            let velocity_y = -1.0;


            let particle_velocity = (velocity_x, velocity_y, velocity_z);
            
            let new_particle = Particle::new(self.position, particle_velocity, self.colour);
     
            particle_system.particles.push(new_particle);
        }

    }

    //particle creation - first in the pipeline
    fn new_spray(&mut self, particle_system: &mut ParticleSystem) {
        let mut rng = rand::thread_rng();
    
        for _i in 0..self.spray_volume {
            let angle = rng.gen_range(-self.spray_angle / 2.0..self.spray_angle / 2.0);
            let velocity_x = rng.gen_range(-1.0..1.0); 
            let velocity_z = rng.gen_range(-1.0..1.0); 
    
            let rotated_velocity_x = velocity_x * angle.cos() - velocity_z * angle.sin();
            let rotated_velocity_z = velocity_x * angle.sin() + velocity_z * angle.cos();
    
            let velocity_y = -1.0; 
    
            let particle_velocity = (rotated_velocity_x, velocity_y, rotated_velocity_z);
    
            let new_particle = Particle::new(self.position, particle_velocity, self.colour);
    
            particle_system.particles.push(new_particle);
        }
    }
}

pub struct ParticleSystem {
    drag: f32,
    particles: Vec<Particle>,
    particles_cv : Arc<(Mutex<Vec<Particle>>,Condvar)>,
    gravity: f32,
}

impl ParticleSystem {
    fn new(drag: f32, gravity: f32) -> ParticleSystem {
        ParticleSystem {
            drag: drag,
            particles: Vec::new(),
            particles_cv : Arc::new((Mutex::new(Vec::new()), Condvar::new())),
            gravity: gravity
        }
    }

    fn CreateParticles(size: usize, colour: (f32, f32, f32)) -> Vec<Particle> {
        let mut particles = vec![];
        for _i in 0..size {
            particles.push(Particle::new((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), colour));
        }
        particles
    }

    fn simple_update(&mut self, delta_time : f32){
        for particle in &mut self.particles {
            
            // Update position
            particle.position.0 += particle.velocity.0 * delta_time;
            particle.position.1 += particle.velocity.1 * delta_time;
            particle.position.2 += particle.velocity.2 * delta_time;
        }
        
    }
    //particle movement - second in the pipeline
    //update particles
    fn update(&mut self, delta_time: f32) {
        for particle in &mut self.particles {
            let velocity_mag = (particle.velocity.0.powi(2)
                + particle.velocity.1.powi(2)
                + particle.velocity.2.powi(2))
            .sqrt();

            let acceleration_mag = self.gravity - self.drag * velocity_mag.powi(2);

            let mut normalized_velocity = particle.velocity;
            let velocity_mag_inv = 1.0 / velocity_mag;
            normalized_velocity.0 *= velocity_mag_inv;
            normalized_velocity.1 *= velocity_mag_inv;
            normalized_velocity.2 *= velocity_mag_inv;

            let acceleration = (
                normalized_velocity.0 * acceleration_mag,
                normalized_velocity.1 * acceleration_mag,
                normalized_velocity.2 * acceleration_mag,
            );

            particle.velocity.0 += acceleration.0 * delta_time;
            particle.velocity.1 += acceleration.1 * delta_time;
            particle.velocity.2 += acceleration.2 * delta_time;

            particle.position.0 += particle.velocity.0 * delta_time;
            particle.position.1 += particle.velocity.1 * delta_time;
            particle.position.2 += particle.velocity.2 * delta_time;
        }
    }


    fn CheckCollisionAgainstPaper(&mut self, paper: &mut Paper) {

        let mut particles_to_remove = vec![];

        for index in 0..self.particles.len() {
            let particle = self.particles[index].clone(); 
            if particle.position.0 < paper.position.0 + (paper.scale.0 / 2.0)
                && particle.position.0 > paper.position.0 - (paper.scale.0 / 2.0)
                && particle.position.1 < paper.position.1 + (paper.scale.1 / 2.0)
                && particle.position.1 > paper.position.1 - (paper.scale.1 / 2.0)
                && particle.position.2 < paper.position.2 + (paper.scale.2 / 2.0)
                && particle.position.2 > paper.position.2 - (paper.scale.2 / 2.0)
            {
                self.HandleCollision(paper, &particle); 
                particles_to_remove.push(index); 
            }
        }

        
        for &index in particles_to_remove.iter().rev() {
            self.particles.remove(index);
        }

    }

    //paper colouring - last in the pipeline
    fn HandleCollision(&mut self, paper: &mut Paper, particle: &Particle) {
        let paper_start_x = paper.position.0 - (paper.scale.0 / 2.0); //X(0) position on paper
        let paper_start_y = paper.position.2 - (paper.scale.2 / 2.0); //Y(0) position on paper

        let x_constant = paper.x_capacity as f32 / paper.scale.0;
        let y_constant = paper.y_capacity as f32 / paper.scale.2;

        let index_x = ((particle.position.0 - paper_start_x) * x_constant) as usize;
        let index_y = ((particle.position.2 - paper_start_y) * y_constant) as usize;

        /*
        paper colour = (1 - b) m + b p

        where

        m is original paper colour
        p is the aerosol colour
        b is the blend coefficient (0.1)
        */

        let b = 0.1;
        
        paper.paint_array[index_x][index_y] = ((1.0 - b) * paper.paint_array[index_x][index_y].0 + b * particle.colour.0,
                                               (1.0 - b) * paper.paint_array[index_x][index_y].1 + b * particle.colour.1,
                                               (1.0 - b) * paper.paint_array[index_x][index_y].2 + b * particle.colour.2);

    }

}

#[derive(Clone)]
struct Particle {
    position: (f32, f32, f32),
    velocity: (f32, f32, f32),
    //colour nums must be between 0 and 1
    colour: (f32, f32, f32),
}

impl Particle {
    fn new(
        position: (f32, f32, f32),
        velocity: (f32, f32, f32),
        color: (f32, f32, f32),
    ) -> Particle {
        Particle {
            position: position,
            velocity: velocity,
            colour: color,
        }
    }
}
